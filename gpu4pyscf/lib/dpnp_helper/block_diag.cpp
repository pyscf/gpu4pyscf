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
#define THREADS        8
// THREADS must be greater than (LMAX+1)*(LMAX+2)/2

__attribute__((always_inline))
static void _block_diag(double *out, int m, int n, double *diags, int ndiags, int *offsets, int *rows, int *cols, sycl::nd_item<2>& item)
{
    int r = item.get_group(1);

    if (r >= ndiags){
        return;
    }
    int m0 = rows[r+1] - rows[r];
    int n0 = cols[r+1] - cols[r];
    
    for (int i = item.get_local_id(1); i < m0; i += THREADS){
        for (int j = item.get_local_id(0); j < n0; j += THREADS){
            out[(i+rows[r])*n + (j+cols[r])] = diags[offsets[r] + i*n0 + j];
        }
    }
}

extern "C" {
int block_diag(sycl::queue& stream, double *out, int m, int n, double *diags, int ndiags, int *offsets, int *rows, int *cols)
{
    sycl::range<2> threads(THREADS, THREADS);
    sycl::range<2> blocks(1, ndiags);
    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { stream._block_diag(out, m, n, diags, ndiags, offsets, rows, cols, item); });
    return 0;
}
}
