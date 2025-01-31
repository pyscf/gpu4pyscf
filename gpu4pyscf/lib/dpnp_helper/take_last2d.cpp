/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
#include <stdio.h>
#define THREADS        32
#define COUNT_BLOCK     80

__attribute__((always_inline))
static void _take_last2d(double *a, const double *b, int *indices, int n. sycl::nd_item<3>& item)
{
    size_t i = item.get_group(0);
    int j = static_cast<int>(item.get_global_id(2));
    int k = static_cast<int>(item.get_global_id(1));
    if (j >= n || k >= n) {
        return;
    }

    int j_b = indices[j];
    int k_b = indices[k];
    int off = i * n * n;

    a[off + j * n + k] = b[off + j_b * n + k_b];
}

__attribute__((always_inline))
static void _takebak(double *out, double *a, int *indices,
                     int count, int n_o, int n_a, sycl::nd_item<2>& item)
{
    int i0 = item.get_group(0) * COUNT_BLOCK;
    int j = static_cast<int>(item.get_global_id(1));
    if (j >= n_a) {
        return;
    }

    // a is on host with zero-copy memory. We need enough iterations for
    // data prefetch to hide latency
    int i1 = i0 + COUNT_BLOCK;
    if (i1 > count) i1 = count;
    int jp = indices[j];
#pragma unroll
    for (size_t i = i0; i < i1; ++i) {
        out[i * n_o + jp] = a[i * n_a + j];
    }
}

extern "C" {
int take_last2d(sycl::queue& stream, double *a, const double *b, int *indices, int blk_size, int n)
{
    // reorder j and k in a[i,j,k] with indicies
    int ntile = (n + THREADS - 1) / THREADS;
    sycl::range<3> threads(1, THREADS, THREADS);
    sycl::range<3> blocks(blk_size, ntile, ntile);
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _take_last2d(a, b, indices, n, item); });
    return 0;
}

int takebak(sycl::queue& stream, double *out, double *a_h, int *indices,
            int count, int n_o, int n_a)
{
    double *a_d = a_h;

    int ntile = (n_a + THREADS*THREADS - 1) / (THREADS*THREADS);
    int ncount = (count + COUNT_BLOCK - 1) / COUNT_BLOCK;
    sycl::range<2> threads(1, THREADS*THREADS);
    sycl::range<2> blocks(ncount, ntile);
    _takebak<<<blocks, threads, 0, stream>>>(out, a_d, indices, count, n_o, n_a);
    return 0;
}
}
