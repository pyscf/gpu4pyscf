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
int cho_solve(cublasHandle_t handle, const double *a, double *b, int m, int n, int lower)
{
    // XA = B
    cublasSideMode_t side = CUBLAS_SIDE_RIGHT; 
    cublasFillMode_t uplo;
    cublasOperation_t trans = CUBLAS_OP_N; 
    cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

    if(lower == 1){
        uplo = CUBLAS_FILL_MODE_UPPER;
    }
    else {
        uplo = CUBLAS_FILL_MODE_LOWER;
    }

    const double alpha = 1.0;

    cublasStatus_t status = cublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, a, n, b, m);

    if(status == CUBLAS_STATUS_SUCCESS){
        return 0;
    }
    return 1;
}
}
