/*
 * gpu4pyscf is a plugin to use Intel GPU in PySCF package
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
void _add_sparse(double *a, double *b, int *indices, int n, int m, int count, sycl::nd_item<3>& item)
{
    int row = item.get_group(2) * BLOCK_DIM + item.get_local_id(2);
    int col = item.get_group(1) * BLOCK_DIM + item.get_local_id(1);
    if (row >= m || col >= m){
        return;
    }
    int idx_a = indices[row] * n + indices[col];
    int idx_b = row * m + col;
    for (int i = 0; i < count; i++){
        a[idx_a + i*n*n] += b[idx_b + i*m*m];
    }
}

extern "C" {
    int add_sparse(sycl::queue stream, double *a, double *b, int *indices, int n, int m, int count){
	int ntile = (m + THREADS - 1) / THREADS;
	sycl::range<3> threads(1, THREADS, THREADS);
	sycl::range<3> blocks(1, ntile, ntile);
	stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _add_sparse(a, b, indices, n, m, count, item); });
	return 0;
    }
}
