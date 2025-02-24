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
#include <oneapi/mkl/blas.hpp>
#define THREADS 32
#define BDIM 32

__attribute__((always_inline))
void _unpack_tril(const double *eri_tril, double *eri, int nao, sycl::nd_item<3>& item){
    int i = static_cast<int>(item.get_global_id(2));
    int j = static_cast<int>(item.get_global_id(1));
    int p = item.get_group(0);
    int stride = ((nao + 1) * nao) / 2;

    if(i >= nao || j >= nao || i < j){
        return;
    }
    int ptr = j + (i+1)*i/2;
    eri[p*nao*nao + j*nao + i] = eri_tril[ptr + p*stride];
}

__attribute__((always_inline))
void _unpack_triu(const double *eri_tril, double *eri, int nao, sycl::nd_item<3>& item){
    int i = static_cast<int>(item.get_global_id(2));
    int j = static_cast<int>(item.get_global_id(1));
    int p = item.get_group(0);
    int stride = ((nao + 1) * nao) / 2;

    if(i >= nao || j >= nao || i > j){
        return;
    }
    int ptr = i + (j+1)*j/2;

    eri[p*nao*nao + j*nao + i] = eri_tril[ptr + p*stride];
}

__attribute__((always_inline))
void _unpack_sparse(const double *cderi_sparse, const long *row, const long *col,
                    double *out, int nao, int nij, int stride_sparse, int p0, int p1, sycl::nd_item<2>& item){
    int ij = static_cast<int>(item.get_global_id(1));
    int k = static_cast<int>(item.get_global_id(0));

    int idx_aux = k + p0;
    if (idx_aux >= p1 || ij >= nij){
        return;
    }

    int i = row[ij];
    int j = col[ij];
    double e = cderi_sparse[ij*stride_sparse + idx_aux];
    out[k + i*(p1-p0) + j*(p1-p0)*nao] = e;
    out[k + j*(p1-p0) + i*(p1-p0)*nao] = e;
}

extern "C" {
int unpack_tril(sycl::queue stream, const double *eri_tril, double *eri, int nao, int blk_size){
    sycl::range<3> threads(1, THREADS, THREADS);
    int nx = (nao + threads[2] - 1) / threads[2];
    int ny = (nao + threads[1] - 1) / threads[1];
    sycl::range<3> blocks(blk_size, ny, nx);
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _unpack_tril(eri_tril, eri, nao, item); });
    stream.parallel_for(sycl::nd_range<3>(blocks * threads, threads), [=](auto item) { _unpack_triu(eri_tril, eri, nao, item); });
    return 0;
}

int unpack_sparse(sycl::queue stream, const double *cderi_sparse, const long *row, const long *col,
                double *eri, int nao, int nij, int naux, int p0, int p1){
    int blockx = (nij + THREADS - 1) / THREADS;
    int blocky = (p1 - p0 + THREADS - 1) / THREADS;
    sycl::range<2> threads(THREADS, THREADS);
    sycl::range<2> blocks(blocky, blockx);

    stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { _unpack_sparse(cderi_sparse, row, col, eri, nao, nij, naux, p0, p1, item); });
    return 0;
}

}
