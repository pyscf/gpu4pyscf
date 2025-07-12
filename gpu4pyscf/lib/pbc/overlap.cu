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

#include "pbc.cuh"
#include "int3c2e.cuh"

#define PI_POW_1_5       5.568327996831707845

typedef struct {
    int8_t iprim;
    int8_t jprim;
    int8_t nfi;
    int8_t nfj;
} PackedPGTO;

#define GOUT_WIDTH      43

static __global__
void int1e_ovlp_kernel(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
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
    int lij = li + lj;
    int stride_j = li + 1;
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
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = gx + gx_len;
    double *gz = gy + gx_len;
    double *rjri = gz + gx_len;
    gx[0] = PI_POW_1_5;
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
                double xjxi = rj[0] + xjL - ri[0];
                double yjyi = rj[1] + yjL - ri[1];
                double zjzi = rj[2] + zjL - ri[2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                rjri[0*nsp_per_block] = xjxi;
                rjri[1*nsp_per_block] = yjyi;
                rjri[2*nsp_per_block] = zjzi;
                rjri[3*nsp_per_block] = rr_ij;
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
                double aj_aij = aj / aij;
                if (gout_id == 0) {
                    double theta = ai * aj_aij;
                    double theta_rr = theta * rjri[3*nsp_per_block];
                    double cicj = ci[ip] * cj[jp];
                    gz[0] = cicj / (aij*sqrt(aij)) * exp(-theta_rr);
                }
                if (lij > 0) {
                    __syncthreads();
                    double s0x, s1x, s2x;
                    double b = .5 / aij;
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = gx + n * gx_len;
                        double xjxi = rjri[n*nsp_per_block];
                        double xpa = xjxi * aj_aij;
                        s0x = _gx[0];
                        s1x = xpa * s0x;
                        _gx[nsp_per_block] = s1x;
                        for (int i = 1; i < lij; ++i) {
                            s2x = xpa * s1x + i * b * s0x;
                            _gx[(i+1)*nsp_per_block] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                        for (int j = 0; j < lj; ++j) {
                            int ij = (lij-j) + j*stride_j;
                            s1x = _gx[ij*nsp_per_block];
                            for (--ij; ij >= j*stride_j; --ij) {
                                s0x = _gx[ij*nsp_per_block];
                                _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                                s1x = s0x;
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

        if (task_id + sp_id < shl_pair1) {
            int *ao_loc = envs.ao_loc;
            int nbas = envs.cell0_nbas;
            int nao = ao_loc[nbas];
            size_t nao2 = nao * nao;
            int cell_id = jsh / nbas;
            int jshp = jsh % nbas;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jshp];
            double *out_subblock = out + cell_id*nao2 + i0 * nao + j0;
            int nfi = pdata.nfi;
            int nfj = pdata.nfj;
            int nfij = nfi * nfj;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                out_subblock[i*nao+j] = gout[n];
            }
        }
    }
}

__global__
void int1e_kin_kernel(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
{
}

__global__
void int1e_ipovlp_kernel(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
{
}

__global__
void int1e_ipkin(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
{
}

extern "C" {
int PBCint1e_ovlp(double *out, PBCIntEnvVars *envs, int shm_size,
                  int nbatches_shl_pair, int *bas_ij_idx,
                  int *shl_pair_offsets, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(int1e_ovlp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    PBCInt2c2eBounds bounds = {
        bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
    };
    int1e_ovlp_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(out, *envs, bounds);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_ovlp kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCint1e_ipovlp(double *out, PBCIntEnvVars *envs, int shm_size,
                    int nbatches_shl_pair, int *bas_ij_idx,
                    int *shl_pair_offsets, int *gout_stride_lookup)
{
    PBCInt2c2eBounds bounds = {
        bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
    };
    int1e_ipovlp_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(out, *envs, bounds);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_ipovlp kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
