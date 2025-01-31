/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <sycl/sycl.hpp>

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

int CPdsymm_triu(double *a, int n, int counts)
{
    int ntile = (n + THREADS - 1) / THREADS;
    sycl::range<3> threads(1, THREADS, THREADS);
    sycl::range<3> blocks(counts, ntile, ntile);
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _dsymm_triu(a, n, item); });
    return 0;
}

int transpose_sum(sycl::queue& stream, double *a, int n, int counts){
    int ntile = (n + THREADS - 1) / THREADS;
    sycl::range<3> threads(1, THREADS, THREADS);
    sycl::range<3> blocks(counts, ntile, ntile);
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _transpose_sum(a, n, item); });
    return 0;
}
}
