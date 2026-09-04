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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_contract_k.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "pbc/recursion.cuh"

#define THREADS         256
#define GOUT_IP_WIDTH   20
#define BLOCK_SIZE      16
#define L_AUX           6
#define L_AUX1          (L_AUX+1)

__global__ static
void e_int2c2e_ip1_kernel(double *out, double *dm, PBCIntEnvVars envs,
                          double omega, double lr_factor, double sr_factor,
                          int *shl_pair_offsets, uint32_t *bas_ij_idx,
                          int *gout_stride_lookup)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int nbas;
    __shared__ int li, lj, nroots, nao, iprim, jprim;
    __shared__ int gout_stride;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        nbas = envs.nbas * envs.bvk_ncells;
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        nroots = (li + lj + 1) / 2 + 1;
        if (omega < 0) {
            nroots *= 2; // omega < 0
        }
        nao = envs.ao_loc[envs.nbas];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
    }
    __syncthreads();
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int stride_j = li + 2;
    int i_1 =          nsp_per_block;
    //int j_1 = stride_j*nsp_per_block;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sp_id;
    double *gx = shared_memory + nsp_per_block * nroots*2 + sp_id;
    double *Rpq = shared_memory + nsp_per_block * (g_size*3+nroots*2) + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        //double v_jx = 0;
        //double v_jy = 0;
        //double v_jz = 0;
        __syncthreads();
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        if (gout_id == 0) {
            double fac = PI_FAC;
            if (ish == jsh) {
                fac *= .5;
            } else if (ish < jsh) {
                fac = 0;
            }
            gx[gx_len] = fac;
        }
        int i0 = envs.ao_loc[ish];
        int j0 = envs.ao_loc[jsh];
        double *dm_local = dm + j0 * nao + i0;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        for (int img = 0; img < envs.nimgs; img++) {
            __syncthreads();
            if (gout_id == 0) {
                double xpq = env[ri+0] - (env[rj+0] + img_coords[img*3+0]);
                double ypq = env[ri+1] - (env[rj+1] + img_coords[img*3+1]);
                double zpq = env[ri+2] - (env[rj+2] + img_coords[img*3+2]);
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                Rpq[0*nsp_per_block] = xpq;
                Rpq[1*nsp_per_block] = ypq;
                Rpq[2*nsp_per_block] = zpq;
                Rpq[3*nsp_per_block] = rr;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ai2 = ai * 2;
                //double aj2 = aj * 2;
                double aij = ai + aj;
                double theta = ai * aj / aij;
                if (gout_id == 0) {
                    double cicj = env[ci+ip] * env[cj+jp];
                    gx[0] = cicj / (ai*aj*sqrt(aij));
                }
                double rr = Rpq[3*nsp_per_block];
                rys_roots_for_k(nroots, theta, rr, rw, omega, lr_factor, sr_factor,
                                nsp_per_block, gout_stride, gout_id);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                    }
                    double rt = rw[ irys*2   *nsp_per_block];
                    vrr(gx, Rpq, ai, aj, rt, li+1, lj, gout_id, gout_stride, nsp_per_block);
                    if (pair_ij < shl_pair1) {
                        float div_nfi = c_div_nf[li];
#pragma unroll
                        for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                            uint32_t j = ij * div_nfi;
                            uint32_t i = ij - nfi * j;
                            int ix = idx_i[i*3+0];
                            int iy = idx_i[i*3+1];
                            int iz = idx_i[i*3+2];
                            int jx = idx_j[j*3+0];
                            int jy = idx_j[j*3+1];
                            int jz = idx_j[j*3+2];
                            int addrx = (ix + jx*stride_j) * nsp_per_block;
                            int addry = (iy + jy*stride_j + g_size) * nsp_per_block;
                            int addrz = (iz + jz*stride_j + g_size*2) * nsp_per_block;
                            double Ix = gx[addrx];
                            double Iy = gx[addry];
                            double Iz = gx[addrz];
                            double dm_ij = dm_local[j*nao+i];
                            double prod_xy = Ix * Iy * dm_ij;
                            double prod_xz = Ix * Iz * dm_ij;
                            double prod_yz = Iy * Iz * dm_ij;
                            double fix = ai2 * gx[addrx+i_1]; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                            double fiy = ai2 * gx[addry+i_1]; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                            double fiz = ai2 * gx[addrz+i_1]; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                            //double fjx = aj2 * gx[addrx+j_1]; if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                            //double fjy = aj2 * gx[addry+j_1]; if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                            //double fjz = aj2 * gx[addrz+j_1]; if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                        }
                    }
                }
            }
        }
        if (pair_ij < shl_pair1) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            atomicAdd(out+ia*3+0, v_ix);
            atomicAdd(out+ia*3+1, v_iy);
            atomicAdd(out+ia*3+2, v_iz);
            atomicAdd(out+ja*3+0, -v_ix);
            atomicAdd(out+ja*3+1, -v_iy);
            atomicAdd(out+ja*3+2, -v_iz);
        }
    }
}

__global__ static
void int2c2e_deriv_kernel(double *de, double *sigma, double *dm, PBCIntEnvVars envs,
                          double omega, double lr_factor, double sr_factor,
                          int *shl_pair_offsets, uint32_t *bas_ij_idx,
                          int *gout_stride_lookup)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int nbas;
    __shared__ int li, lj, nroots, nao, iprim, jprim;
    __shared__ int gout_stride;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        nbas = envs.nbas * envs.bvk_ncells;
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        nroots = (li + lj + 1) / 2 + 1;
        if (omega < 0) {
            nroots *= 2; // omega < 0
        }
        nao = envs.ao_loc[envs.nbas];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
    }
    __syncthreads();
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int stride_j = li + 2;
    int i_1 =          nsp_per_block;
    //int j_1 = stride_j*nsp_per_block;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sp_id;
    double *gx = shared_memory + nsp_per_block * nroots*2 + sp_id;
    double *Rpq = shared_memory + nsp_per_block * (g_size*3+nroots*2) + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);

    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_xz = 0;
    double sigma_yx = 0;
    double sigma_yy = 0;
    double sigma_yz = 0;
    double sigma_zx = 0;
    double sigma_zy = 0;
    double sigma_zz = 0;

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double grad_ix = 0;
        double grad_iy = 0;
        double grad_iz = 0;
        __syncthreads();
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        if (gout_id == 0) {
            double fac = PI_FAC;
            if (ish == jsh) {
                fac *= .5;
            } else if (ish < jsh) {
                fac = 0;
            }
            gx[gx_len] = fac;
        }
        int i0 = envs.ao_loc[ish];
        int j0 = envs.ao_loc[jsh];
        double *dm_local = dm + j0 * nao + i0;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        for (int img = 0; img < envs.nimgs; img++) {
            __syncthreads();
            if (gout_id == 0) {
                double xpq = env[ri+0] - (env[rj+0] + img_coords[img*3+0]);
                double ypq = env[ri+1] - (env[rj+1] + img_coords[img*3+1]);
                double zpq = env[ri+2] - (env[rj+2] + img_coords[img*3+2]);
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                Rpq[0*nsp_per_block] = xpq;
                Rpq[1*nsp_per_block] = ypq;
                Rpq[2*nsp_per_block] = zpq;
                Rpq[3*nsp_per_block] = rr;
            }
            double v_ix = 0;
            double v_iy = 0;
            double v_iz = 0;
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp % iprim;
                int jp = ijp / iprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ai2 = ai * 2;
                //double aj2 = aj * 2;
                double aij = ai + aj;
                double theta = ai * aj / aij;
                if (gout_id == 0) {
                    double cicj = env[ci+ip] * env[cj+jp];
                    gx[0] = cicj / (ai*aj*sqrt(aij));
                }
                double rr = Rpq[3*nsp_per_block];
                rys_roots_for_k(nroots, theta, rr, rw, omega, lr_factor, sr_factor,
                                nsp_per_block, gout_stride, gout_id);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                    }
                    double rt = rw[ irys*2   *nsp_per_block];
                    vrr(gx, Rpq, ai, aj, rt, li+1, lj, gout_id, gout_stride, nsp_per_block);
                    if (pair_ij < shl_pair1) {
                        float div_nfi = c_div_nf[li];
#pragma unroll
                        for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                            uint32_t j = ij * div_nfi;
                            uint32_t i = ij - nfi * j;
                            int ix = idx_i[i*3+0];
                            int iy = idx_i[i*3+1];
                            int iz = idx_i[i*3+2];
                            int jx = idx_j[j*3+0];
                            int jy = idx_j[j*3+1];
                            int jz = idx_j[j*3+2];
                            int addrx = (ix + jx*stride_j) * nsp_per_block;
                            int addry = (iy + jy*stride_j + g_size) * nsp_per_block;
                            int addrz = (iz + jz*stride_j + g_size*2) * nsp_per_block;
                            double Ix = gx[addrx];
                            double Iy = gx[addry];
                            double Iz = gx[addrz];
                            double dm_ij = dm_local[j*nao+i];
                            double prod_xy = Ix * Iy * dm_ij;
                            double prod_xz = Ix * Iz * dm_ij;
                            double prod_yz = Iy * Iz * dm_ij;
                            double fix = ai2 * gx[addrx+i_1]; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                            double fiy = ai2 * gx[addry+i_1]; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                            double fiz = ai2 * gx[addrz+i_1]; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                            //double fjx = aj2 * gx[addrx+j_1]; if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                            //double fjy = aj2 * gx[addry+j_1]; if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                            //double fjz = aj2 * gx[addrz+j_1]; if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                        }
                    }
                }
            }
            double xixj = Rpq[0*nsp_per_block];
            double yiyj = Rpq[1*nsp_per_block];
            double zizj = Rpq[2*nsp_per_block];
            sigma_xx += v_ix * xixj;
            sigma_xy += v_ix * yiyj;
            sigma_xz += v_ix * zizj;
            sigma_yx += v_iy * xixj;
            sigma_yy += v_iy * yiyj;
            sigma_yz += v_iy * zizj;
            sigma_zx += v_iz * xixj;
            sigma_zy += v_iz * yiyj;
            sigma_zz += v_iz * zizj;
            grad_ix += v_ix;
            grad_iy += v_iy;
            grad_iz += v_iz;
        }
        if (pair_ij < shl_pair1) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            double grad_jx = -grad_ix;
            double grad_jy = -grad_iy;
            double grad_jz = -grad_iz;
            atomicAdd(de+ia*3+0, grad_ix);
            atomicAdd(de+ia*3+1, grad_iy);
            atomicAdd(de+ia*3+2, grad_iz);
            atomicAdd(de+ja*3+0, grad_jx);
            atomicAdd(de+ja*3+1, grad_jy);
            atomicAdd(de+ja*3+2, grad_jz);
        }
    }
    atomicAdd(sigma+0, sigma_xx);
    atomicAdd(sigma+1, sigma_xy);
    atomicAdd(sigma+2, sigma_xz);
    atomicAdd(sigma+3, sigma_yx);
    atomicAdd(sigma+4, sigma_yy);
    atomicAdd(sigma+5, sigma_yz);
    atomicAdd(sigma+6, sigma_zx);
    atomicAdd(sigma+7, sigma_zy);
    atomicAdd(sigma+8, sigma_zz);
}

extern "C" {
int e_int2c2e_ip1(double *out, double *dm, PBCIntEnvVars *envs,
                  double omega, double lr_factor, double sr_factor, int shm_size,
                  int nbatches_shl_pair, int *shl_pair_offsets,
                  uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(e_int2c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    e_int2c2e_ip1_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, dm, *envs, omega, lr_factor, sr_factor,
            shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int2c2e_ip1 kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int int2c2e_deriv(double *grad, double *sigma, double *dm, PBCIntEnvVars *envs,
                  double omega, double lr_factor, double sr_factor, int shm_size,
                  int nbatches_shl_pair, int *shl_pair_offsets,
                  uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(int2c2e_deriv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int2c2e_deriv_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            grad, sigma, dm, *envs, omega, lr_factor, sr_factor,
            shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int2c2e_deriv kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
