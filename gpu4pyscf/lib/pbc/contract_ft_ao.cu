/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_contract_k.cuh"

#define WARP_SIZE       32
#define WARPS           8
#define THREADS         256
#define GOUT_WIDTH      30
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787
#define OF_COMPLEX      2
#define POOL_SIZE       65536

__global__ static
void ft_aopair_kernel(double *out, double *vG,
                      PBCIntEnvVars envs, int *shl_pair_offsets,
                      uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                      int *gout_stride_lookup, double *Gv, int nGv,
                      int nbatches_shl_pair, int compressing, int *head)
{
    constexpr int nGv_per_block = 16;
    constexpr unsigned mask = (1u << nGv_per_block) - 1;
    constexpr int sp_threads = THREADS / nGv_per_block;
    constexpr unsigned sp_mask = (1u << sp_threads) - 1;
    int thread_id = threadIdx.x;
    int Gv_id_in_block = thread_id % nGv_per_block;
    int t_id = thread_id / nGv_per_block;
    __shared__ int sp_block_id;
while (1) {
    if (thread_id == 0) {
        sp_block_id = atomicAdd(head, 1);
    }
    __syncthreads();
    if (sp_block_id >= nbatches_shl_pair) {
        return;
    }

    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj;
    __shared__ int iprim, jprim;
    __shared__ int nao;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / bvk_nbas;
        int jsh0 = bas_ij0 - bvk_nbas * ish0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        // Note: must use this ao_loc than envs.ao_loc because envs.ao_loc
        // cannot handle spherical integrals
        nao = envs.ao_loc[envs.nbas];
        gout_stride = gout_stride_lookup[li*LMAX1+lj];
        nsp_per_block = sp_threads / gout_stride;
    }
    __syncthreads();
    int nGsp_per_block = nGv_per_block * nsp_per_block;
    int gout_id = t_id / nsp_per_block;
    int sp_id = t_id - nsp_per_block * gout_id;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int stride_j = li + 1;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nGsp_per_block;
    extern __shared__ double shared_memory[];
    double *gxR = shared_memory + nGv_per_block * sp_id + Gv_id_in_block;
    double *gxI = gxR + gx_len;
    double *gyR = gxR + gx_len*2;
    double *gyI = gxR + gx_len*3;
    double *gzR = gxR + gx_len*4;
    double *gzI = gxR + gx_len*5;
    double *rjri = shared_memory + gx_len*6 + sp_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);

    for (int pair_idx = shl_pair0+sp_id; pair_idx < shl_pair1+sp_id; pair_idx += nsp_per_block) {
        __syncthreads();
        int pair_ij = pair_idx;
        if (pair_idx >= shl_pair1) {
            pair_ij = shl_pair0;
        }
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij % bvk_nbas;
        int img0 = img_offsets[pair_ij];
        int img1 = img_offsets[pair_ij+1];
        __shared__ int img_max;
        __shared__ int img_counts[sp_threads];
        if (Gv_id_in_block == 0) {
            img_counts[t_id] = img1 - img0;
        }
        __syncthreads();
        if (thread_id < sp_threads) {
            int count = img_counts[thread_id];
            for (int offset = sp_threads/2; offset > 0; offset /= 2) {
                count = max(count, __shfl_down_sync(sp_mask, count, offset));
            }
            if (thread_id == 0) {
                img_max = count;
            }
        }
        __syncthreads();

        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double xjxi = rj[0] - xi;
        double yjyi = rj[1] - yi;
        double zjzi = rj[2] - zi;
        double goutR[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            goutR[n] = 0.;
        }

        for (int Gv_id = Gv_id_in_block;
             Gv_id < nGv + Gv_id_in_block; Gv_id += nGv_per_block) {
            double kx = 0;
            double ky = 0;
            double kz = 0;
            if (Gv_id < nGv) {
                kx = Gv[Gv_id];
                ky = Gv[Gv_id + nGv];
                kz = Gv[Gv_id + nGv * 2];
            }
            double kk = kx * kx + ky * ky + kz * kz;
            double s0xR, s1xR, s2xR;
            double s0xI, s1xI, s2xI;
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double a2 = .5 / aij;
                double fac = OVERLAP_FAC * env[ci+ip] * env[cj+jp] / (aij * sqrt(aij));
                for (int img = img0; img < img0+img_max; img++) {
                    __syncthreads();
                    int img_id = 0;
                    if (img < img1) {
                        img_id = img_idx[img];
                    } else {
                        fac = 0;
                    }
                    double Lx = img_coords[img_id*3+0];
                    double Ly = img_coords[img_id*3+1];
                    double Lz = img_coords[img_id*3+2];
                    double xjLxi = xjxi + Lx;
                    double yjLyi = yjyi + Ly;
                    double zjLzi = zjzi + Lz;
                    rjri[0*nsp_per_block] = xjLxi;
                    rjri[1*nsp_per_block] = yjLyi;
                    rjri[2*nsp_per_block] = zjLzi;
                    if (gout_id == 0) {
                        double xij = xjLxi * aj_aij + xi;
                        double yij = yjLyi * aj_aij + yi;
                        double zij = zjLzi * aj_aij + zi;
                        double kR = kx * xij + ky * yij + kz * zij;
                        sincos(-kR, gzI, gzR);
                        double rr = xjLxi*xjLxi + yjLyi*yjLyi + zjLzi*zjLzi;
                        double theta_rr = theta_ij*rr + .5*a2*kk;
                        double Kab = exp(-theta_rr);
                        gxR[0] = fac;
                        gxI[0] = 0.;
                        gyR[0] = vG[Gv_id*OF_COMPLEX  ];
                        gyI[0] = vG[Gv_id*OF_COMPLEX+1];
                        // exp(-theta_rr-kR*1j)
                        gzR[0] *= Kab;
                        gzI[0] *= Kab;
                    }
                    int lij = li + lj;
                    if (lij > 0) {
                        // gx[i+1] = ia2 * gx[i-1] + (rijrx[0] - kx[n]*a2*_Complex_I) * gx[i];
                        __syncthreads();
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gxR = gxR + n * gx_len * OF_COMPLEX;
                            double *_gxI = _gxR + gx_len;
                            double RpaR = rjri[n*nsp_per_block] * aj_aij; // Rp - Ra
                            double RpaI = -a2;
                            if (Gv_id < nGv) {
                                RpaI *= Gv[Gv_id+nGv*n];
                            }
                            s0xR = _gxR[0];
                            s0xI = _gxI[0];
                            s1xR = RpaR * s0xR - RpaI * s0xI;
                            s1xI = RpaR * s0xI + RpaI * s0xR;
                            _gxR[nGsp_per_block] = s1xR;
                            _gxI[nGsp_per_block] = s1xI;
                            for (int i = 1; i < lij; i++) {
                                double ia2 = i * a2;
                                s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
                                s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
                                _gxR[(i+1)*nGsp_per_block] = s2xR;
                                _gxI[(i+1)*nGsp_per_block] = s2xI;
                                s0xR = s1xR;
                                s0xI = s1xI;
                                s1xR = s2xR;
                                s1xI = s2xI;
                            }
                        }
                    }
                    if (lj > 0) {
                        __syncthreads();
                        for (int n = gout_id; n < 3*OF_COMPLEX; n += gout_stride) {
                            double *_gx = gxR + n * gx_len;
                            // The real and imaginary parts call the same expression
                            int _ix = n / 2;
                            double xjxi = rjri[_ix*nsp_per_block];
                            for (int j = 0; j < lj; ++j) {
                                int ij = (lij-j) + j*stride_j;
                                s1xR = _gx[ij*nGsp_per_block];
                                for (--ij; ij >= j*stride_j; --ij) {
                                    s0xR = _gx[ij*nGsp_per_block];
                                    _gx[(ij+stride_j)*nGsp_per_block] = s1xR - xjxi * s0xR;
                                    s1xR = s0xR;
                                }
                            }
                        }
                    }
                    __syncthreads();
                    if (pair_idx < shl_pair1 && img < img1 && Gv_id < nGv) {
                        float div_nfi = c_div_nf[li];
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            uint32_t ij = n*gout_stride + gout_id;
                            if (ij >= nfij) break;
                            uint32_t j = ij * div_nfi;
                            uint32_t i = ij - nfi * j;
                            int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
                            int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
                            int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
                            int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
                            int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
                            int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
                            int addrx = (ix + jx*stride_j) * nGsp_per_block;
                            int addry = (iy + jy*stride_j + g_size*OF_COMPLEX) * nGsp_per_block;
                            int addrz = (iz + jz*stride_j + g_size*2*OF_COMPLEX) * nGsp_per_block;
                            double xR = gxR[addrx];
                            double xI = gxR[addrx+gx_len];
                            double yR = gxR[addry];
                            double yI = gxR[addry+gx_len];
                            double zR = gxR[addrz];
                            double zI = gxR[addrz+gx_len];
                            double xyR = xR * yR - xI * yI;
                            double xyI = xR * yI + xI * yR;
                            goutR[n] += xyR * zR - xyI * zI;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            int ij = n*gout_stride + gout_id;
            if (ij >= nfij) break;
            for (int offset = nGv_per_block / 2; offset > 0; offset /= 2) {
                goutR[n] += __shfl_down_sync(mask, goutR[n], offset);
            }
        }

        if (pair_idx < shl_pair1 && Gv_id_in_block == 0) {
            size_t bvk_Nao = ncells * nao;
            int i0 = envs.ao_loc[ish];
            int cell_id = jsh / envs.nbas;
            int jsh_cell0 = jsh - cell_id * envs.nbas;
            int j0 = envs.ao_loc[jsh_cell0];
            size_t Nao = nao;
            double *aft_tensor = out + (i0 * ncells + cell_id) * Nao + j0;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride + gout_id;
                if (ij >= nfij) break;
                size_t j = ij / nfi;
                size_t i = ij - nfi * j;
                size_t addr = i * bvk_Nao + j;
                atomicAdd(aft_tensor+addr, goutR[n]);
            }
        }
    }
}
}

__global__ static
void ft_pdotp_kernel(double *out, double *vG,
                     PBCIntEnvVars envs, int *shl_pair_offsets,
                     uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                     int *gout_stride_lookup, double *Gv, int nGv,
                     int nbatches_shl_pair, int compressing, int *head)
{
    constexpr int nGv_per_block = 16;
    constexpr unsigned mask = (1u << nGv_per_block) - 1;
    constexpr int sp_threads = THREADS / nGv_per_block;
    constexpr unsigned sp_mask = (1u << sp_threads) - 1;
    int thread_id = threadIdx.x;
    int Gv_id_in_block = thread_id % nGv_per_block;
    int t_id = thread_id / nGv_per_block;
    __shared__ int sp_block_id;
while (1) {
    if (thread_id == 0) {
        sp_block_id = atomicAdd(head, 1);
    }
    __syncthreads();
    if (sp_block_id >= nbatches_shl_pair) {
        return;
    }

    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj;
    __shared__ int iprim, jprim;
    __shared__ int nao;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / bvk_nbas;
        int jsh0 = bas_ij0 - bvk_nbas * ish0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        // Note: must use this ao_loc than envs.ao_loc because envs.ao_loc
        // cannot handle spherical integrals
        nao = envs.ao_loc[envs.nbas];
        gout_stride = gout_stride_lookup[li*LMAX1+lj];
        nsp_per_block = sp_threads / gout_stride;
    }
    __syncthreads();
    int nGsp_per_block = nGv_per_block * nsp_per_block;
    int gout_id = t_id / nsp_per_block;
    int sp_id = t_id - nsp_per_block * gout_id;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int stride_j = li + 2;
    int g_size = stride_j * (lj + 2);
    int gx_len = g_size * nGsp_per_block;
    extern __shared__ double shared_memory[];
    double *gxR = shared_memory + nGv_per_block * sp_id + Gv_id_in_block;
    double *gxI = gxR + gx_len;
    double *gyR = gxR + gx_len*2;
    double *gyI = gxR + gx_len*3;
    double *gzR = gxR + gx_len*4;
    double *gzI = gxR + gx_len*5;
    double *rjri = shared_memory + gx_len*6 + sp_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);

    for (int pair_idx = shl_pair0+sp_id; pair_idx < shl_pair1+sp_id; pair_idx += nsp_per_block) {
        __syncthreads();
        int pair_ij = pair_idx;
        if (pair_idx >= shl_pair1) {
            pair_ij = shl_pair0;
        }
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij % bvk_nbas;
        int img0 = img_offsets[pair_ij];
        int img1 = img_offsets[pair_ij+1];
        __shared__ int img_max;
        __shared__ int img_counts[sp_threads];
        if (Gv_id_in_block == 0) {
            img_counts[t_id] = img1 - img0;
        }
        __syncthreads();
        if (thread_id < sp_threads) {
            int count = img_counts[thread_id];
            for (int offset = sp_threads/2; offset > 0; offset /= 2) {
                count = max(count, __shfl_down_sync(sp_mask, count, offset));
            }
            if (thread_id == 0) {
                img_max = count;
            }
        }
        __syncthreads();

        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double xjxi = rj[0] - xi;
        double yjyi = rj[1] - yi;
        double zjzi = rj[2] - zi;
        double goutR[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            goutR[n] = 0.;
        }

        for (int Gv_id = Gv_id_in_block;
             Gv_id < nGv + Gv_id_in_block; Gv_id += nGv_per_block) {
            double kx = 0;
            double ky = 0;
            double kz = 0;
            if (Gv_id < nGv) {
                kx = Gv[Gv_id];
                ky = Gv[Gv_id + nGv];
                kz = Gv[Gv_id + nGv * 2];
            }
            double kk = kx * kx + ky * ky + kz * kz;
            double s0xR, s1xR, s2xR;
            double s0xI, s1xI, s2xI;
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double a2 = .5 / aij;
                double fac = OVERLAP_FAC * env[ci+ip] * env[cj+jp] / (aij * sqrt(aij));
                for (int img = img0; img < img0+img_max; img++) {
                    __syncthreads();
                    int img_id = 0;
                    if (img < img1) {
                        img_id = img_idx[img];
                    } else {
                        fac = 0;
                    }
                    double Lx = img_coords[img_id*3+0];
                    double Ly = img_coords[img_id*3+1];
                    double Lz = img_coords[img_id*3+2];
                    double xjLxi = xjxi + Lx;
                    double yjLyi = yjyi + Ly;
                    double zjLzi = zjzi + Lz;
                    rjri[0*nsp_per_block] = xjLxi;
                    rjri[1*nsp_per_block] = yjLyi;
                    rjri[2*nsp_per_block] = zjLzi;
                    if (gout_id == 0) {
                        double xij = xjLxi * aj_aij + xi;
                        double yij = yjLyi * aj_aij + yi;
                        double zij = zjLzi * aj_aij + zi;
                        double kR = kx * xij + ky * yij + kz * zij;
                        sincos(-kR, gzI, gzR);
                        double rr = xjLxi*xjLxi + yjLyi*yjLyi + zjLzi*zjLzi;
                        double theta_rr = theta_ij*rr + .5*a2*kk;
                        double Kab = exp(-theta_rr);
                        gxR[0] = fac;
                        gxI[0] = 0.;
                        gyR[0] = vG[Gv_id*OF_COMPLEX  ];
                        gyI[0] = vG[Gv_id*OF_COMPLEX+1];
                        // exp(-theta_rr-kR*1j)
                        gzR[0] *= Kab;
                        gzI[0] *= Kab;
                    }
                    int lij = li + lj + 2;
                    // gx[i+1] = ia2 * gx[i-1] + (rijrx[0] - kx[n]*a2*_Complex_I) * gx[i];
                    __syncthreads();
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gxR = gxR + n * gx_len * OF_COMPLEX;
                        double *_gxI = _gxR + gx_len;
                        double RpaR = rjri[n*nsp_per_block] * aj_aij; // Rp - Ra
                        double RpaI = -a2;
                        if (Gv_id < nGv) {
                            RpaI *= Gv[Gv_id+nGv*n];
                        }
                        s0xR = _gxR[0];
                        s0xI = _gxI[0];
                        s1xR = RpaR * s0xR - RpaI * s0xI;
                        s1xI = RpaR * s0xI + RpaI * s0xR;
                        _gxR[nGsp_per_block] = s1xR;
                        _gxI[nGsp_per_block] = s1xI;
                        for (int i = 1; i < lij; i++) {
                            double ia2 = i * a2;
                            s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
                            s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
                            _gxR[(i+1)*nGsp_per_block] = s2xR;
                            _gxI[(i+1)*nGsp_per_block] = s2xI;
                            s0xR = s1xR;
                            s0xI = s1xI;
                            s1xR = s2xR;
                            s1xI = s2xI;
                        }
                    }
                    __syncthreads();
                    for (int n = gout_id; n < 3*OF_COMPLEX; n += gout_stride) {
                        double *_gx = gxR + n * gx_len;
                        // The real and imaginary parts call the same expression
                        int _ix = n / 2;
                        double xjxi = rjri[_ix*nsp_per_block];
                        for (int j = 0; j <= lj; ++j) {
                            int ij = (lij-j) + j*stride_j;
                            s1xR = _gx[ij*nGsp_per_block];
                            for (--ij; ij >= j*stride_j; --ij) {
                                s0xR = _gx[ij*nGsp_per_block];
                                _gx[(ij+stride_j)*nGsp_per_block] = s1xR - xjxi * s0xR;
                                s1xR = s0xR;
                            }
                        }
                    }
                    __syncthreads();
                    if (pair_idx < shl_pair1 && img < img1 && Gv_id < nGv) {
                        int i_1 =          nGsp_per_block;
                        int j_1 = stride_j*nGsp_per_block;
                        double ai2 = -2. * ai;
                        double aj2 = -2. * aj;
                        float div_nfi = c_div_nf[li];
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            uint32_t ij = n*gout_stride + gout_id;
                            if (ij >= nfij) break;
                            uint32_t j = ij * div_nfi;
                            uint32_t i = ij - nfi * j;
                            int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
                            int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
                            int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
                            int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
                            int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
                            int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
                            int addrx = (ix + jx*stride_j) * nGsp_per_block;
                            int addry = (iy + jy*stride_j + g_size*OF_COMPLEX) * nGsp_per_block;
                            int addrz = (iz + jz*stride_j + g_size*2*OF_COMPLEX) * nGsp_per_block;
                            double xR = gxR[addrx];
                            double xI = gxR[addrx+gx_len];
                            double yR = gxR[addry];
                            double yI = gxR[addry+gx_len];
                            double zR = gxR[addrz];
                            double zI = gxR[addrz+gx_len];
                            double f3xR = ai2 * gxR[addrx+i_1+j_1];
                            double f3xI = ai2 * gxR[addrx+i_1+j_1+gx_len];
                            if (ix > 0) {
                                f3xR += ix * gxR[addrx-i_1+j_1];
                                f3xI += ix * gxR[addrx-i_1+j_1+gx_len];
                            }
                            f3xR *= aj2;
                            f3xI *= aj2;
                            if (jx > 0) {
                                double fxR = ai2 * gxR[addrx+i_1-j_1];
                                double fxI = ai2 * gxR[addrx+i_1-j_1+gx_len];
                                if (ix > 0) {
                                    fxR += ix * gxR[addrx-i_1-j_1];
                                    fxI += ix * gxR[addrx-i_1-j_1+gx_len];
                                }
                                f3xR += jx * fxR;
                                f3xI += jx * fxI;
                            }
                            double yzR = yR * zR - yI * zI;
                            double yzI = yR * zI + yI * zR;
                            goutR[n] += yzR * f3xR - yzI * f3xI;

                            double f3yR = ai2 * gxR[addry+i_1+j_1];
                            double f3yI = ai2 * gxR[addry+i_1+j_1+gx_len];
                            if (iy > 0) {
                                f3yR += iy * gxR[addry-i_1+j_1];
                                f3yI += iy * gxR[addry-i_1+j_1+gx_len];
                            }
                            f3yR *= aj2;
                            f3yI *= aj2;
                            if (jy > 0) {
                                double fyR = ai2 * gxR[addry+i_1-j_1];
                                double fyI = ai2 * gxR[addry+i_1-j_1+gx_len];
                                if (iy > 0) {
                                    fyR += iy * gxR[addry-i_1-j_1];
                                    fyI += iy * gxR[addry-i_1-j_1+gx_len];
                                }
                                f3yR += jy * fyR;
                                f3yI += jy * fyI;
                            }
                            double xzR = xR * zR - xI * zI;
                            double xzI = xR * zI + xI * zR;
                            goutR[n] += xzR * f3yR - xzI * f3yI;

                            double f3zR = ai2 * gxR[addrz+i_1+j_1];
                            double f3zI = ai2 * gxR[addrz+i_1+j_1+gx_len];
                            if (iz > 0) {
                                f3zR += iz * gxR[addrz-i_1+j_1];
                                f3zI += iz * gxR[addrz-i_1+j_1+gx_len];
                            }
                            f3zR *= aj2;
                            f3zI *= aj2;
                            if (jz > 0) {
                                double fzR = ai2 * gxR[addrz+i_1-j_1];
                                double fzI = ai2 * gxR[addrz+i_1-j_1+gx_len];
                                if (iz > 0) {
                                    fzR += iz * gxR[addrz-i_1-j_1];
                                    fzI += iz * gxR[addrz-i_1-j_1+gx_len];
                                }
                                f3zR += jz * fzR;
                                f3zI += jz * fzI;
                            }
                            double xyR = xR * yR - xI * yI;
                            double xyI = xR * yI + xI * yR;
                            goutR[n] += xyR * f3zR - xyI * f3zI;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            int ij = n*gout_stride + gout_id;
            if (ij >= nfij) break;
            for (int offset = nGv_per_block / 2; offset > 0; offset /= 2) {
                goutR[n] += __shfl_down_sync(mask, goutR[n], offset);
            }
        }

        if (pair_idx < shl_pair1 && Gv_id_in_block == 0) {
            size_t bvk_Nao = ncells * nao;
            int i0 = envs.ao_loc[ish];
            int cell_id = jsh / envs.nbas;
            int jsh_cell0 = jsh - cell_id * envs.nbas;
            int j0 = envs.ao_loc[jsh_cell0];
            size_t Nao = nao;
            double *aft_tensor = out + (i0 * ncells + cell_id) * Nao + j0;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride + gout_id;
                if (ij >= nfij) break;
                size_t j = ij / nfi;
                size_t i = ij - nfi * j;
                size_t addr = i * bvk_Nao + j;
                atomicAdd(aft_tensor+addr, goutR[n]);
            }
        }
    }
}
}

extern "C" {
int contract_ft_aopair(double *out, double *vG, PBCIntEnvVars *envs, int *head,
                       int shm_size, int nbatches_shl_pair, int *shl_pair_offsets,
                       uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                       int *gout_stride_lookup, double *grids, int ngrids,
                       int compressing)
{
    cudaFuncSetAttribute(ft_aopair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    cudaMemset(head, 0, sizeof(int));
    ft_aopair_kernel<<<workers, THREADS, shm_size>>>(
        out, vG, *envs, shl_pair_offsets, bas_ij_idx, img_idx, img_offsets,
        gout_stride_lookup, grids, ngrids, nbatches_shl_pair, compressing, head);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int contract_ft_pdotp(double *out, double *vG, PBCIntEnvVars *envs, int *head,
                      int shm_size, int nbatches_shl_pair, int *shl_pair_offsets,
                      uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                      int *gout_stride_lookup, double *grids, int ngrids,
                      int compressing)
{
    cudaFuncSetAttribute(ft_pdotp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    cudaMemset(head, 0, sizeof(int));
    ft_pdotp_kernel<<<workers, THREADS, shm_size>>>(
        out, vG, *envs, shl_pair_offsets, bas_ij_idx, img_idx, img_offsets,
        gout_stride_lookup, grids, ngrids, nbatches_shl_pair, compressing, head);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_pdotp_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
