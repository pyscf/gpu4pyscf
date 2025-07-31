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

#define GOUT_WIDTH      36
#define GOUT_WIDTH_IP1  18

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
    int8_t nfi = (li + 1) * (li + 2) / 2;
    int8_t nfj = (lj + 1) * (lj + 2) / 2;
    int8_t iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int8_t jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    PackedPGTO pdata = {iprim, jprim, nfi, nfj};
    double *env = envs.env;
    double *img_coords = envs.img_coords;

    int gout_stride = bounds.gout_stride_lookup[li*L_AUX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int g_size = (li + 1) * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = gx + gx_len;
    double *gz = gx + gx_len * 2;
    double *rjri = gx + gx_len * 3;
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
            int16_t *idz_ij = idx_ij + nfij * 2;
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
                int lij = li + lj;
                int stride_j = li + 1;
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

static __global__
void int1e_kin_kernel(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
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
    int8_t nfi = (li + 1) * (li + 2) / 2;
    int8_t nfj = (lj + 1) * (lj + 2) / 2;
    int8_t iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int8_t jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    PackedPGTO pdata = {iprim, jprim, nfi, nfj};
    double *env = envs.env;
    double *img_coords = envs.img_coords;

    int gout_stride = bounds.gout_stride_lookup[li*L_AUX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int g_size = (li + 3) * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = gx + gx_len;
    double *gz = gx + gx_len * 2;
    double *rjri = gx + gx_len * 3;
    gx[0] = PI_POW_1_5;
    gy[0] = -.5;

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
            int16_t *idx_i = c_pair_idx + c_pair_offsets[li*L_AUX1];
            int16_t *idy_i = idx_i + nfi;
            int16_t *idz_i = idx_i + nfi * 2;
            int16_t *idx_j = c_pair_idx + c_pair_offsets[lj*L_AUX1];
            int16_t *idy_j = idx_j + nfj;
            int16_t *idz_j = idx_j + nfj * 2;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * -2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                if (gout_id == 0) {
                    double theta = ai * aj_aij;
                    double theta_rr = theta * rjri[3*nsp_per_block];
                    double cicj = ci[ip] * cj[jp];
                    gz[0] = cicj / (aij*sqrt(aij)) * exp(-theta_rr);
                }
                __syncthreads();
                int lij = li + lj + 2;
                int stride_j = li + 3;
                int i_1 = nsp_per_block;
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
                __syncthreads();
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ij = n*gout_stride+gout_id;
                    if (ij >= nfij) break;
                    int i = ij % nfi;
                    int j = ij / nfi;
                    int ix = idx_i[i];
                    int iy = idy_i[i];
                    int iz = idz_i[i];
                    int jx = idx_j[j];
                    int jy = idy_j[j];
                    int jz = idz_j[j];
                    int addrx = (ix + jx*stride_j) * nsp_per_block;
                    int addry = (iy + jy*stride_j) * nsp_per_block;
                    int addrz = (iz + jz*stride_j) * nsp_per_block;
                    double fx0 = gx[addrx];
                    double fy0 = gy[addry];
                    double fz0 = gz[addrz];
                    double fx2 = ai2 * ((ix*2+1)*fx0 + ai2*gx[addrx+i_1*2]);
                    double fy2 = ai2 * ((iy*2+1)*fy0 + ai2*gy[addry+i_1*2]);
                    double fz2 = ai2 * ((iz*2+1)*fz0 + ai2*gz[addrz+i_1*2]);
                    if (ix > 1) fx2 += ix*(ix-1) * gx[addrx-i_1*2];
                    if (iy > 1) fy2 += iy*(iy-1) * gy[addry-i_1*2];
                    if (iz > 1) fz2 += iz*(iz-1) * gz[addrz-i_1*2];
                    gout[n] += fx2 * fy0 * fz0;
                    gout[n] += fx0 * fy2 * fz0;
                    gout[n] += fx0 * fy0 * fz2;
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

static __global__
void int1e_ipovlp_kernel(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
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
    int8_t nfi = (li + 1) * (li + 2) / 2;
    int8_t nfj = (lj + 1) * (lj + 2) / 2;
    int8_t iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int8_t jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    PackedPGTO pdata = {iprim, jprim, nfi, nfj};
    double *env = envs.env;
    double *img_coords = envs.img_coords;

    int gout_stride = bounds.gout_stride_lookup[li*L_AUX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int g_size = (li + 2) * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = gx + gx_len;
    double *gz = gx + gx_len * 2;
    double *rjri = gx + gx_len * 3;
    gx[0] = PI_POW_1_5;
    gy[0] = 1.;

    for (int task_id = shl_pair0; task_id < shl_pair1; task_id += nsp_per_block) {
        double goutx[GOUT_WIDTH_IP1];
        double gouty[GOUT_WIDTH_IP1];
        double goutz[GOUT_WIDTH_IP1];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
            goutx[n] = 0.;
            gouty[n] = 0.;
            goutz[n] = 0.;
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
            int16_t *idx_i = c_pair_idx + c_pair_offsets[li*L_AUX1];
            int16_t *idy_i = idx_i + nfi;
            int16_t *idz_i = idx_i + nfi * 2;
            int16_t *idx_j = c_pair_idx + c_pair_offsets[lj*L_AUX1];
            int16_t *idy_j = idx_j + nfj;
            int16_t *idz_j = idx_j + nfj * 2;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * -2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                if (gout_id == 0) {
                    double theta = ai * aj_aij;
                    double theta_rr = theta * rjri[3*nsp_per_block];
                    double cicj = ci[ip] * cj[jp];
                    gz[0] = cicj / (aij*sqrt(aij)) * exp(-theta_rr);
                }
                int lij = li + lj + 1;
                int stride_j = li + 2;
                int i_1 = nsp_per_block;
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
                __syncthreads();
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                    int ij = n*gout_stride+gout_id;
                    if (ij >= nfij) break;
                    int i = ij % nfi;
                    int j = ij / nfi;
                    int ix = idx_i[i];
                    int iy = idy_i[i];
                    int iz = idz_i[i];
                    int jx = idx_j[j];
                    int jy = idy_j[j];
                    int jz = idz_j[j];
                    int addrx = (ix + jx*stride_j) * nsp_per_block;
                    int addry = (iy + jy*stride_j) * nsp_per_block;
                    int addrz = (iz + jz*stride_j) * nsp_per_block;
                    double fx0 = gx[addrx];
                    double fy0 = gy[addry];
                    double fz0 = gz[addrz];
                    double fx1 = ai2 * gx[addrx+i_1];
                    double fy1 = ai2 * gy[addry+i_1];
                    double fz1 = ai2 * gz[addrz+i_1];
                    if (ix > 0) fx1 += ix * gx[addrx-i_1];
                    if (iy > 0) fy1 += iy * gy[addry-i_1];
                    if (iz > 0) fz1 += iz * gz[addrz-i_1];
                    goutx[n] += fx1 * fy0 * fz0;
                    gouty[n] += fx0 * fy1 * fz0;
                    goutz[n] += fx0 * fy0 * fz1;
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
            double *outx = out + cell_id*nao2*3 + i0 * nao + j0;
            double *outy = outx + nao2;
            double *outz = outx + nao2 * 2;
            int nfi = pdata.nfi;
            int nfj = pdata.nfj;
            int nfij = nfi * nfj;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                outx[i*nao+j] = goutx[n];
                outy[i*nao+j] = gouty[n];
                outz[i*nao+j] = goutz[n];
            }
        }
    }
}

static __global__
void int1e_ipkin_kernel(double *out, PBCIntEnvVars envs, PBCInt2c2eBounds bounds)
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
    int8_t nfi = (li + 1) * (li + 2) / 2;
    int8_t nfj = (lj + 1) * (lj + 2) / 2;
    int8_t iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int8_t jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    PackedPGTO pdata = {iprim, jprim, nfi, nfj};
    double *env = envs.env;
    double *img_coords = envs.img_coords;

    int gout_stride = bounds.gout_stride_lookup[li*L_AUX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int g_size = (li + 4) * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double g[];
    double *gx = g + sp_id;
    double *gy = gx + gx_len;
    double *gz = gx + gx_len * 2;
    double *rjri = gx + gx_len * 3;
    gx[0] = PI_POW_1_5;
    gy[0] = -.5;

    for (int task_id = shl_pair0; task_id < shl_pair1; task_id += nsp_per_block) {
        double goutx[GOUT_WIDTH_IP1];
        double gouty[GOUT_WIDTH_IP1];
        double goutz[GOUT_WIDTH_IP1];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
            goutx[n] = 0.;
            gouty[n] = 0.;
            goutz[n] = 0.;
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
            int16_t *idx_i = c_pair_idx + c_pair_offsets[li*L_AUX1];
            int16_t *idy_i = idx_i + nfi;
            int16_t *idz_i = idx_i + nfi * 2;
            int16_t *idx_j = c_pair_idx + c_pair_offsets[lj*L_AUX1];
            int16_t *idy_j = idx_j + nfj;
            int16_t *idz_j = idx_j + nfj * 2;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * -2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                if (gout_id == 0) {
                    double theta = ai * aj_aij;
                    double theta_rr = theta * rjri[3*nsp_per_block];
                    double cicj = ci[ip] * cj[jp];
                    gz[0] = cicj / (aij*sqrt(aij)) * exp(-theta_rr);
                }
                __syncthreads();
                int lij = li + lj + 3;
                int stride_j = li + 4;
                int i_1 = nsp_per_block;
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
                __syncthreads();
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                    int ij = n*gout_stride+gout_id;
                    if (ij >= nfij) break;
                    int i = ij % nfi;
                    int j = ij / nfi;
                    int ix = idx_i[i];
                    int iy = idy_i[i];
                    int iz = idz_i[i];
                    int jx = idx_j[j];
                    int jy = idy_j[j];
                    int jz = idz_j[j];
                    int addrx = (ix + jx*stride_j) * nsp_per_block;
                    int addry = (iy + jy*stride_j) * nsp_per_block;
                    int addrz = (iz + jz*stride_j) * nsp_per_block;
                    double fx0 = gx[addrx];
                    double fy0 = gy[addry];
                    double fz0 = gz[addrz];
                    double fx1 = ai2 * gx[addrx+i_1];
                    double fy1 = ai2 * gy[addry+i_1];
                    double fz1 = ai2 * gz[addrz+i_1];
                    double fx2 = ai2 * ((ix*2+1)*fx0 + ai2*gx[addrx+i_1*2]);
                    double fy2 = ai2 * ((iy*2+1)*fy0 + ai2*gy[addry+i_1*2]);
                    double fz2 = ai2 * ((iz*2+1)*fz0 + ai2*gz[addrz+i_1*2]);
                    double fx3 = ai2 * ((ix*3+3)*fx1 + ai2*ai2*gx[addrx+i_1*3]);
                    double fy3 = ai2 * ((iy*3+3)*fy1 + ai2*ai2*gy[addry+i_1*3]);
                    double fz3 = ai2 * ((iz*3+3)*fz1 + ai2*ai2*gz[addrz+i_1*3]);
                    if (ix > 0) {
                        double fx1m = ix * gx[addrx-i_1];
                        fx1 += fx1m;
                        fx3 += ai2*(ix*2+1) * fx1m;
                        if (ix > 1) { fx2 += ix*(ix-1)*gx[addrx-i_1*2]; fx3 += ai2*(ix-1)*fx1m; }
                        if (ix > 2) fx3 += ix*(ix-1)*(ix-2) * gx[addrx-i_1*3];
                    }
                    if (iy > 0) {
                        double fy1m = iy * gy[addry-i_1];
                        fy1 += fy1m;
                        fy3 += ai2*(iy*2+1) * fy1m;
                        if (iy > 1) { fy2 += iy*(iy-1)*gy[addry-i_1*2]; fy3 += ai2*(iy-1)*fy1m; }
                        if (iy > 2) fy3 += iy*(iy-1)*(iy-2) * gy[addry-i_1*3];
                    }
                    if (iz > 0) {
                        double fz1m = iz * gz[addrz-i_1];
                        fz1 += fz1m;
                        fz3 += ai2*(iz*2+1) * fz1m;
                        if (iz > 1) { fz2 += iz*(iz-1)*gz[addrz-i_1*2]; fz3 += ai2*(iz-1)*fz1m; }
                        if (iz > 2) fz3 += iz*(iz-1)*(iz-2) * gz[addrz-i_1*3];
                    }
                    goutx[n] += fx3 * fy0 * fz0;
                    goutx[n] += fx1 * fy2 * fz0;
                    goutx[n] += fx1 * fy0 * fz2;
                    gouty[n] += fx2 * fy1 * fz0;
                    gouty[n] += fx0 * fy3 * fz0;
                    gouty[n] += fx0 * fy1 * fz2;
                    goutz[n] += fx2 * fy0 * fz1;
                    goutz[n] += fx0 * fy2 * fz1;
                    goutz[n] += fx0 * fy0 * fz3;
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
            double *outx = out + cell_id*nao2*3 + i0 * nao + j0;
            double *outy = outx + nao2;
            double *outz = outx + nao2 * 2;
            int nfi = pdata.nfi;
            int nfj = pdata.nfj;
            int nfij = nfi * nfj;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH_IP1; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                int j = ij / nfi;
                int i = ij % nfi;
                outx[i*nao+j] = goutx[n];
                outy[i*nao+j] = gouty[n];
                outz[i*nao+j] = goutz[n];
            }
        }
    }
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

int PBCint1e_kin(double *out, PBCIntEnvVars *envs, int shm_size,
                 int nbatches_shl_pair, int *bas_ij_idx,
                 int *shl_pair_offsets, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(int1e_kin_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    PBCInt2c2eBounds bounds = {
        bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
    };
    int1e_kin_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(out, *envs, bounds);
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
    cudaFuncSetAttribute(int1e_ipovlp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
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

int PBCint1e_ipkin(double *out, PBCIntEnvVars *envs, int shm_size,
                    int nbatches_shl_pair, int *bas_ij_idx,
                    int *shl_pair_offsets, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(int1e_ipkin_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    PBCInt2c2eBounds bounds = {
        bas_ij_idx, shl_pair_offsets, gout_stride_lookup,
    };
    int1e_ipkin_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(out, *envs, bounds);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_ipkin kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
