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
#include "vhf.cuh"
#include "rys_roots.cu"
#include "rys_contract_k.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"

#define THREADS         256
#define BLOCK_SIZE      16
#define L_AUX           6
#define L_AUX1          (L_AUX+1)

__global__ static
void e_int2c2e_ip2_kernel(double *out, double *dm, PBCIntEnvVars envs,
                          double omega, double lr_factor, double sr_factor,
                          int *shl_pair_offsets, uint32_t *bas_ij_idx,
                          int *gout_stride_lookup)
{
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    int nbas = envs.nbas;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, nroots;
    __shared__ int iprim, jprim;
    __shared__ int g_size;
    __shared__ int nao;
    __shared__ int gout_stride, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 % nbas;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        int lij = li + lj + 2;
        nroots = lij/ 2 + 1;
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[nbas];
        int stride_j = li + 3;
        g_size = stride_j * (lj + 3);
        gout_stride = gout_stride_lookup[li*L_AUX1+lj];
        nsp_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sp_id;
    double *gx = shared_memory + nsp_per_block * nroots*2 + sp_id;
    double *Rpq = shared_memory + nsp_per_block * (g_size*3+nroots*2) + sp_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_ixjx = 0;
        double v_ixjy = 0;
        double v_ixjz = 0;
        double v_iyjx = 0;
        double v_iyjy = 0;
        double v_iyjz = 0;
        double v_izjx = 0;
        double v_izjy = 0;
        double v_izjz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int i0 = envs.ao_loc[ish];
        int j0 = envs.ao_loc[jsh];
        double *dm_local = dm + j0 * nao + i0;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (gout_id == 0) {
            double xpq = env[ri+0] - env[rj+0];
            double ypq = env[ri+1] - env[rj+1];
            double zpq = env[ri+2] - env[rj+2];
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            Rpq[0*nsp_per_block] = xpq;
            Rpq[1*nsp_per_block] = ypq;
            Rpq[2*nsp_per_block] = zpq;
            Rpq[3*nsp_per_block] = rr;
            double fac = PI_FAC;
            if (ish == jsh) {
                fac *= .5;
            } else if (ish < jsh) {
                fac = 0;
            }
            gx[gx_len] = fac;
        }
        for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
            __syncthreads();
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = env[expi+ip];
            double aj = env[expj+jp];
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
                int stride_j = li + 3;
                int i_1 =          nsp_per_block;
                int j_1 = stride_j*nsp_per_block;
                int nsp = nsp_per_block;
                __syncthreads();
                if (gout_id == 0) {
                    gx[gx_len*2] = rw[(irys*2+1)*nsp];
                }
                double rt = rw[ irys*2   *nsp];
                double rt_aa = rt / aij;
                __syncthreads();
                double rt_ai = rt_aa * aj;
                double b10 = .5/ai * (1 - rt_ai);
                double s0x, s1x, s2x;
                for (int n = gout_id; n < 3; n += gout_stride) {
                    double *_gx = gx + n * gx_len;
                    double c0x = -rt_ai * Rpq[n*nsp];
                    s0x = _gx[0];
                    s1x = c0x * s0x;
                    _gx[nsp] = s1x;
                    for (int i = 1; i < li+2; ++i) {
                        s2x = c0x * s1x + i * b10 * s0x;
                        _gx[(i+1)*nsp] = s2x;
                        s0x = s1x;
                        s1x = s2x;
                    }
                }
                int li3 = (li+3)*3;
                double rt_ak  = rt_aa * ai;
                double b00 = .5 * rt_aa;
                double b01 = .5/aj  * (1 - rt_ak);
                for (int n = gout_id; n < li3+gout_id; n += gout_stride) {
                    __syncthreads();
                    int i = n / 3;
                    int _ix = n % 3;
                    double *_gx = gx + (i + _ix * g_size) * nsp;
                    double cpx = rt_ak * Rpq[_ix*nsp];
                    if (n < li3) {
                        s0x = _gx[0];
                        s1x = cpx * s0x;
                        if (i > 0) {
                            s1x += i * b00 * _gx[-nsp];
                        }
                        _gx[stride_j*nsp] = s1x;
                    }
                    for (int j = 1; j < lj+2; ++j) {
                        __syncthreads();
                        if (n < li3) {
                            s2x = cpx*s1x + j*b01*s0x;
                            if (i > 0) {
                                s2x += i * b00 * _gx[(j*stride_j-1)*nsp];
                            }
                            _gx[(j*stride_j+stride_j)*nsp] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }
                }
                __syncthreads();
                if (pair_ij < shl_pair1) {
                    int nfi = c_nf[li];
                    int nfj = c_nf[lj];
                    int nfij = nfi * nfj;
                    float div_nfi = c_div_nf[li];
                    double ai2 = ai * 2;
                    double aj2 = aj * 2;
#pragma unroll
                    for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                        uint32_t j = ij * div_nfi;
                        uint32_t i = ij - j * nfi;
                        int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                        int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                        int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                        int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                        int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                        int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                        int addrx = (ix + jx*stride_j) * nsp;
                        int addry = (iy + jy*stride_j + g_size) * nsp;
                        int addrz = (iz + jz*stride_j + g_size*2) * nsp;
                        double Ix = gx[addrx];
                        double Iy = gx[addry];
                        double Iz = gx[addrz];
                        double dm_ij = dm_local[j*nao+i];
                        double Ix_d = Ix * dm_ij;
                        double Iy_d = Iy * dm_ij;
                        double Iz_d = Iz * dm_ij;
                        double prod_yz = Iy * Iz_d;
                        double prod_xz = Ix * Iz_d;
                        double prod_xy = Ix * Iy_d;
                        double gix = gx[addrx+i_1];
                        double giy = gx[addry+i_1];
                        double giz = gx[addrz+i_1];
                        double gjx = gx[addrx+j_1];
                        double gjy = gx[addry+j_1];
                        double gjz = gx[addrz+j_1];
                        double fix = ai2 * gix;
                        double fiy = ai2 * giy;
                        double fiz = ai2 * giz;
                        double fjx = aj2 * gjx;
                        double fjy = aj2 * gjy;
                        double fjz = aj2 * gjz;
                        if (ix > 0) { fix -= ix * gx[addrx-i_1]; }
                        if (iy > 0) { fiy -= iy * gx[addry-i_1]; }
                        if (iz > 0) { fiz -= iz * gx[addrz-i_1]; }
                        if (jx > 0) { fjx -= jx * gx[addrx-j_1]; }
                        if (jy > 0) { fjy -= jy * gx[addry-j_1]; }
                        if (jz > 0) { fjz -= jz * gx[addrz-j_1]; }

                        double gijx = gx[addrx+i_1+j_1];
                        double gijy = gx[addry+i_1+j_1];
                        double gijz = gx[addrz+i_1+j_1];
                        double f3x = ai2 * gijx;
                        double f3y = ai2 * gijy;
                        double f3z = ai2 * gijz;
                        if (ix > 0) { f3x -= ix * gx[addrx-i_1+j_1]; }
                        if (iy > 0) { f3y -= iy * gx[addry-i_1+j_1]; }
                        if (iz > 0) { f3z -= iz * gx[addrz-i_1+j_1]; }
                        f3x *= aj2;
                        f3y *= aj2;
                        f3z *= aj2;
                        if (jx > 0) {
                            double fx = ai2 * gx[addrx+i_1-j_1];
                            if (ix > 0) { fx -= ix * gx[addrx-i_1-j_1]; }
                            f3x -= jx * fx;
                        }
                        if (jy > 0) {
                            double fy = ai2 * gx[addry+i_1-j_1];
                            if (iy > 0) { fy -= iy * gx[addry-i_1-j_1]; }
                            f3y -= jy * fy;
                        }
                        if (jz > 0) {
                            double fz = ai2 * gx[addrz+i_1-j_1];
                            if (iz > 0) { fz -= iz * gx[addrz-i_1-j_1]; }
                            f3z -= jz * fz;
                        }
                        v_ixjx += f3x * prod_yz;
                        v_iyjy += f3y * prod_xz;
                        v_izjz += f3z * prod_xy;
                        v_ixjy += fix * fjy * Iz_d;
                        v_ixjz += fix * fjz * Iy_d;
                        v_iyjx += fiy * fjx * Iz_d;
                        v_iyjz += fiy * fjz * Ix_d;
                        v_izjx += fiz * fjx * Iy_d;
                        v_izjy += fiz * fjy * Ix_d;

                        f3x = ai2 * (ai2 * gx[addrx+i_1*2] - (2*ix+1) * Ix);
                        f3y = ai2 * (ai2 * gx[addry+i_1*2] - (2*iy+1) * Iy);
                        f3z = ai2 * (ai2 * gx[addrz+i_1*2] - (2*iz+1) * Iz);
                        if (ix > 1) { f3x += ix*(ix-1) * gx[addrx-i_1*2]; }
                        if (iy > 1) { f3y += iy*(iy-1) * gx[addry-i_1*2]; }
                        if (iz > 1) { f3z += iz*(iz-1) * gx[addrz-i_1*2]; }
                        v_ixx += f3x * prod_yz;
                        v_iyy += f3y * prod_xz;
                        v_izz += f3z * prod_xy;
                        v_ixy += fix * fiy * Iz_d;
                        v_ixz += fix * fiz * Iy_d;
                        v_iyz += fiy * fiz * Ix_d;

                        f3x = aj2 * (aj2 * gx[addrx+j_1*2] - (2*jx+1) * Ix);
                        f3y = aj2 * (aj2 * gx[addry+j_1*2] - (2*jy+1) * Iy);
                        f3z = aj2 * (aj2 * gx[addrz+j_1*2] - (2*jz+1) * Iz);
                        if (jx > 1) { f3x += jx*(jx-1) * gx[addrx-j_1*2]; }
                        if (jy > 1) { f3y += jy*(jy-1) * gx[addry-j_1*2]; }
                        if (jz > 1) { f3z += jz*(jz-1) * gx[addrz-j_1*2]; }
                        v_jxx += f3x * prod_yz;
                        v_jyy += f3y * prod_xz;
                        v_jzz += f3z * prod_xy;
                        v_jxy += fjx * fjy * Iz_d;
                        v_jxz += fjx * fjz * Iy_d;
                        v_jyz += fjy * fjz * Ix_d;
                    }
                }
            }
        }
        if (pair_ij < shl_pair1) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            int natm = envs.natm;
            atomicAdd(out + (ia*natm+ja)*9 + 0, v_ixjx);
            atomicAdd(out + (ia*natm+ja)*9 + 1, v_ixjy);
            atomicAdd(out + (ia*natm+ja)*9 + 2, v_ixjz);
            atomicAdd(out + (ia*natm+ja)*9 + 3, v_iyjx);
            atomicAdd(out + (ia*natm+ja)*9 + 4, v_iyjy);
            atomicAdd(out + (ia*natm+ja)*9 + 5, v_iyjz);
            atomicAdd(out + (ia*natm+ja)*9 + 6, v_izjx);
            atomicAdd(out + (ia*natm+ja)*9 + 7, v_izjy);
            atomicAdd(out + (ia*natm+ja)*9 + 8, v_izjz);
            atomicAdd(out + (ia*natm+ia)*9 + 0, v_ixx * .5);
            atomicAdd(out + (ia*natm+ia)*9 + 3, v_ixy);
            atomicAdd(out + (ia*natm+ia)*9 + 4, v_iyy * .5);
            atomicAdd(out + (ia*natm+ia)*9 + 6, v_ixz);
            atomicAdd(out + (ia*natm+ia)*9 + 7, v_iyz);
            atomicAdd(out + (ia*natm+ia)*9 + 8, v_izz * .5);
            atomicAdd(out + (ja*natm+ja)*9 + 0, v_jxx * .5);
            atomicAdd(out + (ja*natm+ja)*9 + 3, v_jxy);
            atomicAdd(out + (ja*natm+ja)*9 + 4, v_jyy * .5);
            atomicAdd(out + (ja*natm+ja)*9 + 6, v_jxz);
            atomicAdd(out + (ja*natm+ja)*9 + 7, v_jyz);
            atomicAdd(out + (ja*natm+ja)*9 + 8, v_jzz * .5);
        }
    }
}

extern "C" {
int e_int2c2e_ip2(double *out, double *dm, PBCIntEnvVars *envs,
                  double omega, double lr_factor, double sr_factor, int shm_size,
                  int nbatches_shl_pair, int *shl_pair_offsets,
                  uint32_t *bas_ij_idx, int *gout_stride_lookup)
{
    cudaFuncSetAttribute(e_int2c2e_ip2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    e_int2c2e_ip2_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            out, dm, *envs, omega, lr_factor, sr_factor,
            shl_pair_offsets, bas_ij_idx, gout_stride_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int2c2e_ip2 kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
