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

#define THREADS         256
#define BLOCK_SIZE      16

__global__ static
void ejk_int3c2e_ip1_kernel(double *ejk, double *dm, double *density_auxvec,
                            RysIntEnvVars envs, int *shl_pair_offsets,
                            uint32_t *bas_ij_idx, int *ksh_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int *batch_aux_offsets, int naux)
{
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lij, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    //__shared__ int stride_j, stride_k, g_size;
    __shared__ int nfi, nfij, nf;
    __shared__ int nao;
    __shared__ int gout_stride;
    __shared__ double omega;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 - nbas * ish0;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        nksh = ksh1 - ksh0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        lij = li + lj + 1;
        nroots = (lij + lk) / 2 + 1;
        omega = env[PTR_RANGE_OMEGA];
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[nbas];
        nfi = (li + 1) * (li + 2) / 2;
        int nfj = (lj + 1) * (lj + 2) / 2;
        int nfk = (lk + 1) * (lk + 2) / 2;
        nfij = nfi * nfj;
        nf = nfij * nfk;
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    }
    __syncthreads();
    int nst_per_block = THREADS / gout_stride;
    int aux_per_block = min(nst_per_block, BLOCK_SIZE);
    int nsp_per_block = nst_per_block / aux_per_block;
    register int gout_id = thread_id / nst_per_block;
    register int st_id = thread_id - gout_id * nst_per_block;
    register int sp_id = st_id / aux_per_block;
    register int aux_id = st_id - sp_id * aux_per_block;

    int stride_j = li + 2;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 2);
    int i_1 =          nst_per_block;
    int j_1 = stride_j*nst_per_block;
    int k_1 = stride_k*nst_per_block;
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(lk);

    if (gout_id == 0) {
        Rpq[0*nst_per_block] = 0;
        Rpq[1*nst_per_block] = 0;
        Rpq[2*nst_per_block] = 0;
        // Rpq[3] must be initialized. An uninitialized Rpq[3] might be nan,
        // which would cause illegal addresses in the rys_roots function
        Rpq[3*nst_per_block] = 0;
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
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
        double fac = PI_FAC;
        if (ish == jsh) {
            fac *= .5;
        } else if (ish < jsh) {
            fac = 0;
        }
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        if (gout_id == 0 && aux_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = batch_aux_offsets[ksh_block_id] + ksh - ksh_offsets[ksh_block_id];
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                k0 = envs.ao_loc[ksh] - envs.ao_loc[nbas];
                dm_tensor = dm + j0 * nao + i0;
            }

            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                if (gout_id == 0) {
                    double theta_ij = ai * aj_aij;
                    double xjxi = rjri[0*nsp_per_block];
                    double yjyi = rjri[1*nsp_per_block];
                    double zjzi = rjri[2*nsp_per_block];
                    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    double Kab = theta_ij * rr_ij;
                    double fac_ij = exp(-Kab);
                    double xij = xjxi * aj_aij + ri[0];
                    double yij = yjyi * aj_aij + ri[1];
                    double zij = zjzi * aj_aij + ri[2];
                    double xk = rk[0];
                    double yk = rk[1];
                    double zk = rk[2];
                    double xpq = xij - xk;
                    double ypq = yij - yk;
                    double zpq = zij - zk;
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                    Rpq[3*nst_per_block] = rr;
                    double cicj = fac * ci[ip] * cj[jp];
                    gx[gx_len] = cicj * fac_ij;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = expk[kp];
                    double ak2 = ak * 2;
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = ck[kp] / (aij*ak*sqrt(aij+ak));
                    }
                    //TODO: rys_roots_for_k
                    rys_roots_rs(nroots, theta, Rpq[3*nst_per_block], omega,
                                 rw, nst_per_block, gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nst_per_block];
                        }
                        double rt = rw[ irys*2   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double s0x, s1x, s2x;
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
                            double Rpa = rjri[n*nsp_per_block] * aj_aij;
                            //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                            double c0x = Rpa - rt_aij * Rpq[n*nst_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nst_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nst_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                        int lij3 = (lij+1)*3;
                        double rt_ak  = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/ak  * (1 - rt_ak );
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n - i*3; // TODO: remove _ix for nroots > 2
                            double *_gx = gx + (i + _ix * g_size) * nst_per_block;
                            double cpx = rt_ak * Rpq[_ix*nst_per_block];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nst_per_block];
                                }
                                _gx[stride_k*nst_per_block] = s1x;
                            }
                            for (int k = 1; k <= lk; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nst_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nst_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }

                        if (lj > 0) {
                            __syncthreads();
                            if (pair_ij < shl_pair1 && kidx < ksh1) {
                                int lk3 = (lk+2)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m - k*3;
                                    double xjxi = rjri[_ix*nsp_per_block];
                                    double *_gx = gx + (_ix*g_size + k*stride_k) * nst_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nst_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nst_per_block];
                                            _gx[(ij+stride_j)*nst_per_block] = s1x - xjxi * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            for (int n = gout_id; n < nf; n+=gout_stride) {
                                int k  = n / nfij;
                                int ij = n - nfij * k;
                                int j = ij / nfi;
                                int i = ij - nfi * j;
                                int ix = idx_i[i*3+0];
                                int iy = idx_i[i*3+1];
                                int iz = idx_i[i*3+2];
                                int jx = idx_j[j*3+0];
                                int jy = idx_j[j*3+1];
                                int jz = idx_j[j*3+2];
                                int kx = idx_k[k*3+0];
                                int ky = idx_k[k*3+1];
                                int kz = idx_k[k*3+2];
                                double dm_ijk;
                                if (density_auxvec == NULL) {
                                    dm_ijk = dm_tensor[ij*naux + k*nksh];
                                } else {
                                    dm_ijk = dm_tensor[j*nao+i] * density_auxvec[k0+k];
                                }
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nst_per_block;
                                int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nst_per_block;
                                int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nst_per_block;
                                double Ix = gx[addrx];
                                double Iy = gx[addry];
                                double Iz = gx[addrz];
                                double prod_xy = Ix * Iy * dm_ijk;
                                double prod_xz = Ix * Iz * dm_ijk;
                                double prod_yz = Iy * Iz * dm_ijk;
                                double gix = gx[addrx+i_1];
                                double giy = gx[addry+i_1];
                                double giz = gx[addrz+i_1];
                                double gkx = gx[addrx+k_1];
                                double gky = gx[addry+k_1];
                                double gkz = gx[addrz+k_1];
                                double fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                                double fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                                double fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                                double fkx = ak2 * gkx; if (kx > 0) { fkx -= kx * gx[addrx-k_1]; } v_kx += fkx * prod_yz;
                                double fky = ak2 * gky; if (ky > 0) { fky -= ky * gx[addry-k_1]; } v_ky += fky * prod_xz;
                                double fkz = ak2 * gkz; if (kz > 0) { fkz -= kz * gx[addrz-k_1]; } v_kz += fkz * prod_xy;
                                double fjx = aj2 * (gix - rjri[0*nst_per_block] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                                double fjy = aj2 * (giy - rjri[1*nst_per_block] * Iy); if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                                double fjz = aj2 * (giz - rjri[2*nst_per_block] * Iz); if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                            }
                        }
                    }
                }
            }
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
            double *reduce = shared_memory + thread_id;
            __syncthreads();
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                reduce[0*THREADS] = v_kx * 2;
                reduce[1*THREADS] = v_ky * 2;
                reduce[2*THREADS] = v_kz * 2;
            }
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i && pair_ij < shl_pair1 && kidx < ksh1) {
#pragma unroll
                    for (int n = 0; n < 3; ++n) {
                        reduce[n*THREADS] += reduce[n*THREADS+i*nst_per_block];
                    }
                }
            }
            if (gout_id == 0 && pair_ij < shl_pair1 && kidx < ksh1) {
                atomicAdd(ejk+ka*3+0, reduce[0*THREADS]);
                atomicAdd(ejk+ka*3+1, reduce[1*THREADS]);
                atomicAdd(ejk+ka*3+2, reduce[2*THREADS]);
            }
        }
        if (pair_ij < shl_pair1) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

extern "C" {
int ejk_int3c2e_ip1(double *ejk, double *dm, double *density_auxvec,
                    RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int *batch_aux_offsets, int naux)
{
    cudaFuncSetAttribute(ejk_int3c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    ejk_int3c2e_ip1_kernel<<<blocks, THREADS, shm_size>>>(
            ejk, dm, density_auxvec, *envs, shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, batch_aux_offsets, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
