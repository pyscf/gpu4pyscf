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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "contract_rho.cu"

#define THREADS        32
#define BLOCK_DIM   32

__global__
void _add_sparse(double *a, double *b, int *indices, int n, int m, int count)
{
	int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int col = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (row >= m || col >= m){
        return;
    }
    int idx_a = indices[row] * n + indices[col];
    int idx_b = row * m + col;
    for (int i = 0; i < count; i++){
        a[idx_a + i*n*n] += b[idx_b + i*m*m];
    }
}


extern "C"{
__host__
int add_sparse(cudaStream_t stream, double *a, double *b, int *indices, int n, int m, int count){
    int ntile = (m + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(ntile, ntile);
    _add_sparse<<<blocks, threads, 0, stream>>>(a, b, indices, n, m, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

__host__
int GDFTeval_rho2(cudaStream_t stream, cublasHandle_t handle,
                const double* ao, const double* mocc, int ngrids, int nocc, int nao,
                const int xctype_code, double* buf1, double* buf2, double* rho)
{
    // mocc: nao x nocc
    // ao: nao x ngrids
    // rho: 1 x ngrids for LDA
    //      4 x ngrids for GGA
    //      5 x ngrids for mGGA

    double alpha, beta, contract_factor;
    // LDA
    alpha = 1.0;
    beta = 0.0;
    // buf1 = np.dot(mocc.T, ao) in C-order
    // buf1 = np.dot(ao.T, mocc) in F-order
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                ngrids, nocc, nao,
                &alpha,
                ao, ngrids,
                mocc, nocc,
                &beta,
                buf1, ngrids);
    GDFTcontract_rho(stream, rho, buf1, buf1, 0.0, 1.0, ngrids, nocc);

    if (xctype_code > 0) {
        // GGA
        alpha = 1.0;
        beta = 0.0;
        contract_factor = 0.0;
        for (int i = 1; i < 4; i++){
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        ngrids, nocc, nao,
                        &alpha,
                        ao + i*nao*ngrids, ngrids,
                        mocc, nocc,
                        &beta,
                        buf2, ngrids);
            GDFTcontract_rho(stream, rho + i*ngrids, buf1, buf2, 0.0, 2.0, ngrids, nocc);

            if (xctype_code > 1) {
                // mGGA
                // rho[4] += 0.5 * np.dot(buf2.T, buf2)
                GDFTcontract_rho(stream, rho + 4*ngrids, buf2, buf2, contract_factor, 0.5, ngrids, nocc);
                contract_factor = 1.0; // Add to the previous result
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTscale_ao: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int GDFTeval_vxc(cudaStream_t stream, cublasHandle_t handle, int xctype_code,
                const double* ao, const double* wv, double *buf1, double *buf2,
                int ngrids, int nao_mask, int nao, int* idx, double* vmat){
    /*
        Calculate vxc matrix, and add the matrix block to the global vxc matrix
    */

    int err;

    if (xctype_code == 0){
        err = GDFTscale_ao(stream, buf1, ao, wv, ngrids, nao_mask, 1);
    } else {
        err = GDFTscale_ao(stream, buf1, ao, wv, ngrids, nao_mask, 4);
    }
    // LDA
    // buf2 = ao.dot(buf1.T)
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nao_mask, nao_mask, ngrids,
                &alpha,
                ao, ngrids,
                buf1, ngrids,
                &beta,
                buf2, nao_mask);

    // mGGA
    // buf2 += 0.5 * ao[0].dot((wv * ao[0]).T)
    //       + 0.5 * ao[1].dot((wv * ao[1]).T)
    //       + 0.5 * ao[2].dot((wv * ao[2]).T)
    if (xctype_code > 1){
        alpha = 0.5;
        beta = 1.0;
        for (int i = 1; i < 4; i++){
            const double *ao_i = ao + i*nao_mask*ngrids;
            err = GDFTscale_ao(stream, buf1, ao_i, wv+4*ngrids, ngrids, nao_mask, 1);
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        nao_mask, nao_mask, ngrids,
                        &alpha,
                        ao_i, ngrids,
                        buf1, ngrids,
                        &beta,
                        buf2, nao_mask);
        }
    }

    err = add_sparse(stream, vmat, buf2, idx, nao, nao_mask, 1);

    return err;
}

}
