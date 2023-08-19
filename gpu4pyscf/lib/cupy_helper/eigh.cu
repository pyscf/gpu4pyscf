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

#include <cuda_runtime.h>
#include <cusolverDn.h>

extern "C" {
__host__
int eigh(cusolverDnHandle_t cusolverH, double *a, double *b, double *w, int n)
{
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;     // A*x = (lambda)*B*x
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverStatus_t status;

    int lda = n;
    int lwork = 0;

    int *info;
    double *work = nullptr;
    
    // cache it?
    cudaMalloc(reinterpret_cast<void **>(&info), sizeof(int));
    status = cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo, n, a, lda, b, lda, w, &lwork);
    if(status != CUSOLVER_STATUS_SUCCESS){
        return 1;
    }
    cudaMalloc(reinterpret_cast<void **>(&work), sizeof(double) * lwork);
    status = cusolverDnDsygvd(cusolverH, itype, jobz, uplo, n, a, lda, b, lda, w, work, lwork, info);
    if(status != CUSOLVER_STATUS_SUCCESS){
        return 1;
    }
     /* free resources */
    cudaFree(info);
    cudaFree(work);

    return 0;
}
}
