/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/rys_roots.cu"
#include "pbc.cuh"
#include "int3c2e.cuh"

typedef struct {
    int8_t iprim;
    int8_t jprim;
    int8_t nfi;
    int8_t nfj;
} PackedPGTO;

#define GOUT_WIDTH      43

__global__
void pbc_int2c2e_kernel(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int *bas = envs.bas;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    int nroots = (li + lj) / 2 + 1;
    nroots *= 2; // omega < 0
    int8_t nfi = (li + 1) * (li + 2) / 2;
    int8_t nfj = (lj + 1) * (lj + 2) / 2;
    int8_t iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int8_t jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    PackedPGTO pdata = {iprim, jprim, nfi, nfj};
    double *env = envs.env;
    double *img_coords = envs.img_coords;

    int gout_stride = bounds.gout_stride_lookup[lj*L_AUX1+li];
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int g_size = (li + 1) * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + sp_id;
    double *g = rw + nsp_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + gx_len;
    double *gz = gy + gx_len;
    double *Rpq = gz + gx_len;
    gy[0] = 1.;

    for (int task_id = shl_pair0; task_id < shl_pair1; task_id += nsp_per_block) {
        double gout[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            gout[n] = 0.;
        }
        int pair_ij = task_id + sp_id;
        if (pair_ij >= shl_pair1) {
            pair_ij = shl_pair0;
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        for (int img = 0; img < envs.nimgs; img++) {
            if (gout_id == 0) {
                double xjL = img_coords[img*3+0];
                double yjL = img_coords[img*3+1];
                double zjL = img_coords[img*3+2];
                double xpq = ri[0] - (rj[0] + xjL);
                double ypq = ri[1] - (rj[1] + yjL);
                double zpq = ri[2] - (rj[2] + zjL);
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                Rpq[0*nsp_per_block] = xpq;
                Rpq[1*nsp_per_block] = ypq;
                Rpq[2*nsp_per_block] = zpq;
                Rpq[3*nsp_per_block] = rr;
            }
            int iprim = pdata.iprim;
            int jprim = pdata.jprim;
            int ijprim = iprim * jprim;
            int nfi = pdata.nfi;
            int nfj = pdata.nfj;
            int nfij = nfi * nfj;
            int16_t *idx_ij = c_pair_idx + c_pair_offsets[li*L_AUX1+lj];
            int16_t *idy_ij = idx_ij + nfij;
            int16_t *idz_ij = idy_ij + nfij;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double theta = ai * aj / aij;
                if (gout_id == 0) {
                    double cicj = ci[ip] * cj[jp];
                    gx[0] = PI_FAC * cicj / (ai*aj*sqrt(aij));
                }

                double omega = env[PTR_RANGE_OMEGA];
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * Rpq[3*nsp_per_block];
                int _nroots = nroots / 2;
                rys_roots(_nroots, theta_rr, rw+nroots*nsp_per_block,
                          nsp_per_block, gout_id, gout_stride);
                rys_roots(_nroots, theta_fac*theta_rr, rw,
                          nsp_per_block, gout_id, gout_stride);
                __syncthreads();
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                    rw[ irys*2   *nsp_per_block] *= theta_fac;
                    rw[(irys*2+1)*nsp_per_block] *= sqrt_theta_fac;
                }
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gz[0] = rw[(irys*2+1)*nsp_per_block];
                    }
                    double rt = rw[ irys*2   *nsp_per_block];
                    double rt_aa = rt / aij;

                    if (li > 0) {
                        __syncthreads();
                        double rt_ai = rt_aa * aj;
                        double b10 = .5/ai * (1 - rt_ai);
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
                            double c0x = -rt_ai * Rpq[n*nsp_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nsp_per_block] = s1x;
                            for (int i = 1; i < li; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nsp_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    if (lj > 0) {
                        int li3 = (li+1)*3;
                        int stride_j = li + 1;
                        int g_size = stride_j * (lj + 1);
                        double rt_ak  = rt_aa * ai;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/aj  * (1 - rt_ak);
                        for (int n = gout_id; n < li3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(li+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                            double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                            //for i in range(li+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < li3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsp_per_block];
                                }
                                _gx[stride_j*nsp_per_block] = s1x;
                            }
                            for (int j = 1; j < lj; ++j) {
                                __syncthreads();
                                if (n < li3) {
                                    s2x = cpx*s1x + j*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(j*stride_j-1)*nsp_per_block];
                                    }
                                    _gx[(j*stride_j+stride_j)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }
                    }

                    __syncthreads();
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        int ij = n*gout_stride+gout_id;
                        if (ij >= nfij) break;
                        int addrx = idx_ij[ij] * nsp_per_block;
                        int addry = idy_ij[ij] * nsp_per_block;
                        int addrz = idz_ij[ij] * nsp_per_block;
                        gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                    }
                }
            }
        }

        if (task_id + sp_id < shl_pair1) {
            int *ao_loc = envs.ao_loc;
            int nbas = envs.cell0_nbas;
            int nao = ao_loc[nbas];
            size_t nao2 = nao * nao;
            int cell_id = jsh / nbas;
            int jshp = jsh % nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jshp];
            double *eri_tensor = out + cell_id*nao2 + i0 * nao + j0;
            int nfi = pdata.nfi;
            int nfj = pdata.nfj;
            int nfij = nfi * nfj;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                eri_tensor[i*nao+j] = gout[n];
            }
        }
    }
}

__global__ static
void aopair_fill_triu_kernel(double *out, int *conj_mapping, int bvk_ncells, int nao)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nao || j >= nao || i <= j) {
        return;
    }
    size_t nao2 = nao * nao;
    size_t ij = i * nao + j;
    size_t ji = j * nao + i;
    for (int k = 0; k < bvk_ncells; ++k) {
        int ck = conj_mapping[k];
        out[ji + ck*nao2] = out[ij + k*nao2];
    }
}

extern "C" {
int aopair_fill_triu(double *out, int *conj_mapping, int nao, int bvk_ncells)
{
    dim3 threads(16, 16);
    int nao_b = (nao + 15) / 16;
    dim3 blocks(nao_b, nao_b);
    aopair_fill_triu_kernel<<<blocks, threads>>>(out, conj_mapping, bvk_ncells, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in aopair_fill_triu: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
