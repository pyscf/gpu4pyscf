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
#include "contract_rho.cuh"
#include "gsycl/gpu_compat.h"
// Tree reduction along iy dimension in shared memory buf.
#define REDUCE_Y(buf, ixy, iy) \
    for (int _s_ = BLKSIZEY >> 1; _s_ > 0; _s_ >>= 1) { \
        if ((iy) < _s_) { \
            (buf)[(ixy)] += (buf)[(ixy) + BLKSIZEX * _s_]; \
        } \
        __syncthreads(); \
    }
static_assert((BLKSIZEY & (BLKSIZEY - 1)) == 0, "BLKSIZEY must be a power of 2");

// TODO: improve this?
__global__
void GDFTcontract_rho_kernel(double *rho, const double *bra, const double *ket, int ngrids, int nao)
{
    setup_context();
    int grid_id = global_x;
    SHARED_ARRAY(double, buf, [BLKSIZEX*(BLKSIZEY+1)]);
    int ix = threadIdx_x;
    int iy = threadIdx_y;

    const bool active = grid_id < ngrids;
    size_t Ngrids = ngrids;
    double v = 0;
    if (active){
        for (int ao_id = threadIdx_y; ao_id < nao; ao_id += BLKSIZEY) {
            int ket_idx = grid_id + ao_id * Ngrids;
            v += (bra[ket_idx] * ket[ket_idx]);
        }
    }

    int ixy = ix + BLKSIZEX * iy;
    buf[ixy] = v;   __syncthreads();
    REDUCE_Y(buf, ixy, iy);

    if (iy == 0 && active) {
        rho[grid_id] = buf[ix];
    }
}

// half of the GGA rho
__global__
void GDFTcontract_rho4_kernel(double *rho, double *bra, double *ket, int ngrids, int nao, int count)
{
    setup_context();
    int grid_id = global_x;
    SHARED_ARRAY(double, buf, [BLKSIZEX*(BLKSIZEY+1)]);
    int ix = threadIdx_x;
    int iy = threadIdx_y;
    const bool active = grid_id < ngrids;
    size_t ket_stride = nao * ngrids;
    size_t rho_stride = count * ngrids;

    for (int ia = 0; ia < count; ia++){
        double v[4] = {0.0, 0.0, 0.0, 0.0};
        if (active){
            for (int ao_id = threadIdx_y; ao_id < nao; ao_id += BLKSIZEY) {
                int ket_idx = grid_id + ao_id * ngrids;
                double bra_tmp = bra[ket_idx + ia * ket_stride];
                v[0] += bra_tmp * ket[0*ket_stride + ket_idx];
                v[1] += bra_tmp * ket[1*ket_stride + ket_idx];
                v[2] += bra_tmp * ket[2*ket_stride + ket_idx];
                v[3] += bra_tmp * ket[3*ket_stride + ket_idx];
            }
        }

        int ixy = ix + BLKSIZEX * iy;
        for (int i = 0; i < 4; i++){
            buf[ixy] = v[i];   __syncthreads();
            REDUCE_Y(buf, ixy, iy);

            if (iy == 0 && active) {
                rho[grid_id + ia * ngrids + rho_stride * i] = buf[ix];
            }
        }
    }
}

__global__
void GDFTcontract_rho_gga_kernel(double *rho, double *bra, double *ket, int ngrids, int nao)
{
    setup_context();
    const int grid_id = global_x;
    SHARED_ARRAY(double, buf, [BLKSIZEX*(BLKSIZEY+1)]);
    const int ix = threadIdx_x;
    const int iy = threadIdx_y;
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    size_t ket_stride = nao * ngrids;

    double v[4] = {0.0, 0.0, 0.0, 0.0};
    if (active){
        for (int ao_id = threadIdx_y; ao_id < nao; ao_id += BLKSIZEY) {
            size_t ket_idx = grid_id + ao_id * Ngrids;
            double bra_tmp = bra[ket_idx];
            double ket_tmp = ket[ket_idx];

            v[0] += bra_tmp * ket_tmp;

            ket_idx += ket_stride;
            v[1] += bra_tmp * ket[ket_idx];
            v[1] += ket_tmp * bra[ket_idx];

            ket_idx += ket_stride;
            v[2] += bra_tmp * ket[ket_idx];
            v[2] += ket_tmp * bra[ket_idx];

            ket_idx += ket_stride;
            v[3] += bra_tmp * ket[ket_idx];
            v[3] += ket_tmp * bra[ket_idx];
        }
    }

    int ixy = ix + BLKSIZEX * iy;

    for (int i = 0; i < 4; i++){
        buf[ixy] = v[i];   __syncthreads();
        REDUCE_Y(buf, ixy, iy);

        if (iy == 0 && active) {
            rho[grid_id + ngrids * i] = buf[ix];
        }
    }
}


__global__
void GDFTcontract_rho_mgga_kernel(double *rho, double *bra, double *ket, int ngrids, int nao)
{
    setup_context();
    const int grid_id = global_x;
    SHARED_ARRAY(double, buf, [BLKSIZEX*(BLKSIZEY+1)]);
    const int ix = threadIdx_x;
    const int iy = threadIdx_y;
    const bool active = grid_id < ngrids;

    size_t Ngrids = ngrids;
    size_t ket_stride = nao * ngrids;

    double v[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (active){
        for (int ao_id = threadIdx_y; ao_id < nao; ao_id += BLKSIZEY) {
            size_t ket_idx = grid_id + ao_id * Ngrids;
            double bra_tmp0 = bra[ket_idx];
            double ket_tmp0 = ket[ket_idx];

            v[0] += bra_tmp0 * ket_tmp0;

            ket_idx += ket_stride;
            double bra_tmp1 = bra[ket_idx];
            double ket_tmp1 = ket[ket_idx];
            v[1] += bra_tmp0 * ket_tmp1;
            v[1] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

            ket_idx += ket_stride;
            bra_tmp1 = bra[ket_idx];
            ket_tmp1 = ket[ket_idx];
            v[2] += bra_tmp0 * ket_tmp1;
            v[2] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

            ket_idx += ket_stride;
            bra_tmp1 = bra[ket_idx];
            ket_tmp1 = ket[ket_idx];
            v[3] += bra_tmp0 * ket_tmp1;
            v[3] += ket_tmp0 * bra_tmp1;
            v[4] += bra_tmp1 * ket_tmp1;

        }
    }

    v[4] *= 0.5;

    int ixy = ix + BLKSIZEX * iy;

    for (int i = 0; i < 5; i++){
        buf[ixy] = v[i];   __syncthreads();
        REDUCE_Y(buf, ixy, iy);

        if (iy == 0 && active) {
            rho[grid_id + ngrids * i] = buf[ix];
        }
    }
}

static __global__
void dscale_ao_kernel(double *out, double *ket, double *wv,
                      int ngrids, int nao, int nvar)
{
    setup_context();
    int grid_id = global_x;
    int ao_id = global_y;
    if (grid_id >= ngrids || ao_id >= nao) {
        return;
    }

    size_t Ngrids = ngrids;
    size_t Nag = nao * Ngrids;
    size_t ixy = grid_id + ao_id * Ngrids;
    double val = 0;
    int n;
    for (n = 0; n < nvar; ++n) {
        val += ket[ixy + Nag * n] * wv[grid_id + ngrids * n];
    }
    out[ixy] = val;
}

static __global__
void zscale_ao_kernel(double *out, double *ket, double *wv,
                      int ngrids, int nao, int nvar)
{
    setup_context();
    int grid_id = global_x;
    int ao_id = global_y;
    if (grid_id >= ngrids || ao_id >= nao) {
        return;
    }

    size_t Ngrids = ngrids;
    size_t Nag = nao * Ngrids;
    size_t ixy = grid_id + ao_id * Ngrids;
    double vR = 0;
    double vI = 0;
    int n;
    for (n = 0; n < nvar; ++n) {
        size_t ket_off = ixy + Nag * n;
        size_t wv_off = grid_id + ngrids * n;
        double aR = ket[ket_off*2+0];
        double aI = ket[ket_off*2+1];
        double bR = wv[wv_off*2+0];
        double bI = wv[wv_off*2+1];
        vR += aR * bR - aI * bI;
        vI += aR * bI + aI * bR;
    }
    out[ixy*2+0] = vR;
    out[ixy*2+1] = vI;
}

__global__
void GDFT_make_dR_dao_w_kernel(double *out, double *ket, double *wv,
                               int ngrids, int nao)
{
    setup_context();
    int grid_id = global_x;
    int ao_id = global_y;
    if (grid_id >= ngrids || ao_id >= nao) {
        return;
    }

    size_t Ngrids = ngrids;
    size_t Nag = nao * Ngrids;
    size_t ixy = grid_id + ao_id * Ngrids;

    double wv0 = wv[grid_id + ngrids * 0];
    double wv1 = wv[grid_id + ngrids * 1];
    double wv2 = wv[grid_id + ngrids * 2];
    double wv3 = wv[grid_id + ngrids * 3];

    double ket5 = ket[ixy + Nag * 5];
    double ket6 = ket[ixy + Nag * 6];
    double val;
    val = ket[ixy + Nag * 1] * wv0;
    val+= ket[ixy + Nag * 4] * wv1;
    val+= ket5 * wv2;
    val+= ket6 * wv3;
    out[ixy + Nag * 0] = val;

    double ket8 = ket[ixy + Nag * 8];
    val = ket[ixy + Nag * 2] * wv0;
    val+= ket5 * wv1;
    val+= ket[ixy + Nag * 7] * wv2;
    val+= ket8 * wv3;
    out[ixy + Nag * 1] = val;

    val = ket[ixy + Nag * 3] * wv0;
    val+= ket6 * wv1;
    val+= ket8 * wv2;
    val+= ket[ixy + Nag * 9] * wv3;
    out[ixy + Nag * 2] = val;
}


extern "C"{
__host__
int GDFTcontract_rho(cudaStream_t stream, double *rho, const double *bra, const double *ket, int ngrids, int nao)
{
    auto threads = make_block(BLKSIZEX, BLKSIZEY);
    auto blocks = make_grid((ngrids+BLKSIZEX-1)/BLKSIZEX);
    LAUNCH_KERNEL(GDFTcontract_rho_kernel, blocks, threads, 0, stream,
                  rho, bra, ket, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTcontract_rho4(cudaStream_t stream, double *rho, double *bra, double *ket, int ngrids, int nao, int count)
{
    auto threads = make_block(BLKSIZEX, BLKSIZEY);
    auto blocks = make_grid((ngrids+BLKSIZEX-1)/BLKSIZEX);
    LAUNCH_KERNEL(GDFTcontract_rho4_kernel, blocks, threads, 0, stream,
                  rho, bra, ket, ngrids, nao, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTcontract_rho_gga(cudaStream_t stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    auto threads = make_block(BLKSIZEX, BLKSIZEY);
    auto blocks = make_grid((ngrids+BLKSIZEX-1)/BLKSIZEX);
    LAUNCH_KERNEL(GDFTcontract_rho_gga_kernel, blocks, threads, 0, stream,
                  rho, bra, ket, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho_gga: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTcontract_rho_mgga(cudaStream_t stream, double *rho, double *bra, double *ket, int ngrids, int nao)
{
    auto threads = make_block(BLKSIZEX, BLKSIZEY);
    auto blocks = make_grid((ngrids+BLKSIZEX-1)/BLKSIZEX);
    LAUNCH_KERNEL(GDFTcontract_rho_mgga_kernel, blocks, threads, 0, stream,
                  rho, bra, ket, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTcontract_rho_mgga: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFT_make_dR_dao_w(cudaStream_t stream, double *out, double *ket, double *wv,
                 int ngrids, int nao)
{
    auto threads = make_block(BLKSIZEX, BLKSIZEY);
    auto blocks = make_grid((ngrids+BLKSIZEX-1)/BLKSIZEX, (nao+BLKSIZEY-1)/BLKSIZEY);
    LAUNCH_KERNEL(GDFT_make_dR_dao_w_kernel, blocks, threads, 0, stream,
                  out, ket, wv, ngrids, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFT_make_dR_dao_w: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GDFTscale_ao(double *out, double *ket, double *wv,
                 int ngrids, int nao, int nvar, int is_real)
{
    auto threads = make_block(BLKSIZEX, BLKSIZEY);
    auto blocks = make_grid((ngrids+BLKSIZEX-1)/BLKSIZEX, (nao+BLKSIZEY-1)/BLKSIZEY);
    if (is_real) {
        LAUNCH_KERNEL_Q(sycl_get_queue(), dscale_ao_kernel, blocks, threads, 0,
                        out, ket, wv, ngrids, nao, nvar);
    } else {
        LAUNCH_KERNEL_Q(sycl_get_queue(), zscale_ao_kernel, blocks, threads, 0,
                        out, ket, wv, ngrids, nao, nvar);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTscale_ao: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}
