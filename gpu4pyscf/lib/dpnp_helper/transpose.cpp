/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gint/sycl_device.hpp"

#define THREADS 32
#define BLOCK_DIM 32

__attribute__((always_inline))
static void _dsymm_triu(double *a, int n, sycl::nd_item<3>& item)
{
    int i = item.get_global_id(2);
    int j = item.get_global_id(1);
    if (i < j || i >= n || j >= n) {
        return;
    }
    size_t N = n;
    size_t off = N * N * item.get_group(0);
    a[off + j * N + i] = a[off + i * N + j];
}

__attribute__((always_inline))
void _transpose_sum(double *a, int n, sycl::nd_item<3>& item)
{
    if(item.get_group(2) > item.get_group(1)){
        return;
    }
    sycl::group thread_block = item.get_group();
    using tile_t = double[BLOCK_DIM][BLOCK_DIM + 1];
    tile_t& block = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(thread_block);

    unsigned int blockx_off = item.get_group(2) * BLOCK_DIM;
    unsigned int blocky_off = item.get_group(1) * BLOCK_DIM;
    unsigned int x0 = blockx_off + item.get_local_id(1);
    unsigned int y0 = blocky_off + item.get_local_id(0);
    unsigned int x1 = blocky_off + item.get_local_id(1);
    unsigned int y1 = blockx_off + item.get_local_id(0);
    unsigned int z = item.get_group(0);

    unsigned int off = n * n * z;
    unsigned int xy0 = y0 * n + x0 + off;
    unsigned int xy1 = y1 * n + x1 + off;

    if (x0 < n && y0 < n){
        block[item.get_local_id(0)][item.get_local_id(1)] = a[xy0];
    }
    sycl::group_barrier(thread_block);
    if (x1 < n && y1 < n){
        block[item.get_local_id(1)][item.get_local_id(0)] += a[xy1];
    }
    sycl::group_barrier(thread_block);

    if(x0 < n && y0 < n){
        a[xy0] = block[item.get_local_id(0)][item.get_local_id(1)];
    }
    if(x1 < n && y1 < n){
        a[xy1] = block[item.get_local_id(1)][item.get_local_id(0)];
    }
}

extern "C" {

int CPdsymm_triu(sycl::queue stream, double *a, int n, int counts)
{
    int ntile = (n + THREADS - 1) / THREADS;
    sycl::range<3> threads(1, THREADS, THREADS);
    sycl::range<3> blocks(counts, ntile, ntile);
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _dsymm_triu(a, n, item); });
    return 0;
}

int transpose_sum(sycl::queue stream, double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
    sycl::range<3> threads(1, THREADS, THREADS);
    sycl::range<3> blocks(counts, ntile, ntile);
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _transpose_sum(a, n, item); });
    return 0;
}
}
