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
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "unrolled_ejk_int3c2e_ip1.cu"

#define THREADS         256
#define BLOCK_SIZE      16
#define DM_BLOCK        4
#define GOUT_WIDTH      54

__global__ static
void ejk_int3c2e_ip1_kernel(double *ejk, double *ejk_aux,
                            double *dm, double *density_auxvec, int n_dm,
                            RysIntEnvVars envs, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                            int *ksh_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int aux_offset, int npairs, int naux
                            #ifdef USE_SYCL
                            , sycl::nd_item<2> &item, char *shm_mem
                            #endif
                            )
{
    #ifdef USE_SYCL
    int threadIdx_x = item.get_local_id(1);
    int blockIdx_x = item.get_group(1);
    int blockIdx_y = item.get_group(0);
    int gridDim_x = item.get_group_range(1);
    int gridDim_y = item.get_group_range(0);

    auto thread_block = item.get_group();
    int &shl_pair0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &shl_pair1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ksh0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ksh1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nksh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nf = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &g_size = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nao = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nst_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &aux_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nsp_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);

    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    #else
    int threadIdx_x = threadIdx.x;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int gridDim_x = gridDim.x;
    int gridDim_y = gridDim.y;

    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lij, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ int nao;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nsp_per_block;
    extern __shared__ double shared_memory[];
    #endif
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx_x;
    int sp_block_id = gridDim_x - blockIdx_x - 1;
    int ksh_block_id = gridDim_y - blockIdx_y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];

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
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[nbas];
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        int stride_j = li + 2;
        int stride_k = stride_j * (lj + 1);
        g_size = stride_k * (lk + 2);
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
        aux_per_block = min(nst_per_block, BLOCK_SIZE);
        nsp_per_block = nst_per_block / aux_per_block;
    }
    __syncthreads();
    if (n_dm == 1 &&
        int3c2e_ip1_unrolled(ejk, ejk_aux, dm, density_auxvec, envs,
            shl_pair0, shl_pair1, ksh0, ksh1,
            iprim, jprim, kprim, li, lj, lk, bas_ij_idx,
            ao_pair_loc, aux_offset, naux, nao, shared_memory)) {
        return;
    }
    register int gout_id = thread_id / nst_per_block;
    register int st_id = thread_id - gout_id * nst_per_block;
    register int sp_id = st_id / aux_per_block;
    register int aux_id = st_id - sp_id * aux_per_block;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int gx_len = g_size * nst_per_block;

    double *rjri = shared_memory;
    double *Rpq = shared_memory + nsp_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

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
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (gout_id == 0 && aux_id == 0) {
            double xjxi = env[rj+0] - env[ri+0];
            double yjyi = env[rj+1] - env[ri+1];
            double zjzi = env[rj+2] - env[ri+2];
            rjri[sp_id+0*nsp_per_block] = xjxi;
            rjri[sp_id+1*nsp_per_block] = yjyi;
            rjri[sp_id+2*nsp_per_block] = zjzi;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double dm_tensor[GOUT_WIDTH];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    float div_nfi = c_div_nf[li];
                    float div_nfj = c_div_nf[lj];
                    float div_nfij = div_nfi * div_nfj;
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        uint32_t ijk = n*gout_stride+gout_id;
                        if (ijk >= nf) break;
                        uint32_t k = ijk * div_nfij;
                        uint32_t ij = ijk - k * nfi*nfj;
                        dm_tensor[n] = dm_local[ij*naux + k*nksh];
                    }
                } else {
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        dm_tensor[n] = 0;
                    }
                    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
                        int i0 = envs.ao_loc[ish];
                        int j0 = envs.ao_loc[jsh];
                        int k0 = envs.ao_loc[ksh] - nao;
                        double *dm_local = dm + (i_dm * nao + j0) * (size_t)nao + i0;
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            uint32_t ijk = n*gout_stride+gout_id;
                            if (ijk >= nf) break;
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            uint32_t jk = ijk * div_nfi;
                            uint32_t i = ijk - jk * nfi;
                            uint32_t k = jk * div_nfj;
                            uint32_t j = jk - k * nfj;
                            dm_tensor[n] += dm_local[j*nao+i] * density_auxvec[i_dm*naux+k0+k];
                        }
                    }
                }
            }

            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                __syncthreads();
                if (gout_id == 0) {
                    double theta_ij = ai * aj_aij;
                    double xjxi = rjri[sp_id+0*nsp_per_block];
                    double yjyi = rjri[sp_id+1*nsp_per_block];
                    double zjzi = rjri[sp_id+2*nsp_per_block];
                    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    double Kab = theta_ij * rr_ij;
                    double fac_ij = PI_FAC;
                    if (ish == jsh) {
                        fac_ij *= .5;
                    } else if (ish < jsh) {
                        fac_ij = 0;
                    }
                    double cicj = fac_ij * env[ci+ip] * env[cj+jp];
                    gx[gx_len] = cicj * exp(-Kab);
                    double xij = xjxi * aj_aij + env[ri+0];
                    double yij = yjyi * aj_aij + env[ri+1];
                    double zij = zjzi * aj_aij + env[ri+2];
                    double xk = env[rk+0];
                    double yk = env[rk+1];
                    double zk = env[rk+2];
                    double xpq = xij - xk;
                    double ypq = yij - yk;
                    double zpq = zij - zk;
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                    Rpq[3*nst_per_block] = rr;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = env[expk+kp];
                    double ak2 = ak * 2;
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                    }
                    //TODO: rys_roots_for_k
                    rys_roots_rs(nroots, theta, Rpq[3*nst_per_block], omega,
                                 rw, nst_per_block, gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int nsp = nsp_per_block;
                        int nst = nst_per_block;
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 1);
                        int gsize = g_size;
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nst];
                        }
                        double rt = rw[ irys*2   *nst];
                        double rt_aa = rt / (aij + ak);
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double s0x, s1x, s2x;
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
                            double Rpa = rjri[sp_id+n*nsp] * aj_aij;
                            //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                            double c0x = Rpa - rt_aij * Rpq[n*nst];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nst] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nst] = s2x;
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
                            double *_gx = gx + (i + _ix * gsize) * nst;
                            double cpx = rt_ak * Rpq[_ix*nst];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nst];
                                }
                                _gx[stride_k*nst] = s1x;
                            }
                            for (int k = 1; k <= lk; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nst];
                                    }
                                    _gx[(k*stride_k+stride_k)*nst] = s2x;
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
                                    double xjxi = rjri[sp_id+_ix*nsp];
                                    double *_gx = gx + (_ix*gsize + k*stride_k) * nst;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nst];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nst];
                                            _gx[(ij+stride_j)*nst] = s1x - xjxi * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            int i_1 =          nst;
                            int j_1 = stride_j*nst;
                            int k_1 = stride_k*nst;
                            double ai2 = ai * 2;
                            double aj2 = aj * 2;
#pragma unroll
                            for (int n = 0; n < GOUT_WIDTH; ++n) {
                                uint32_t ijk = n*gout_stride+gout_id;
                                if (ijk >= nf) break;
                                uint32_t jk = ijk * div_nfi;
                                uint32_t i = ijk - jk * nfi;
                                uint32_t k = jk * div_nfj;
                                uint32_t j = jk - k * nfj;
                                int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                                int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                                int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                                int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                                int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                                int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                                int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
                                int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
                                int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
                                double dm_ijk = dm_tensor[n];
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                                int addry = (iy + jy*stride_j + ky*stride_k + gsize) * nst;
                                int addrz = (iz + jz*stride_j + kz*stride_k + gsize*2) * nst;
                                double Ix = gx[addrx];
                                double Iy = gx[addry];
                                double Iz = gx[addrz];
                                double prod_xy = Ix * Iy * dm_ijk;
                                double prod_xz = Ix * Iz * dm_ijk;
                                double prod_yz = Iy * Iz * dm_ijk;
                                double gix = gx[addrx+i_1];
                                double giy = gx[addry+i_1];
                                double giz = gx[addrz+i_1];
                                double fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                                double fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                                double fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                                double fjx = aj2 * (gix - rjri[sp_id+0*nsp] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                                double fjy = aj2 * (giy - rjri[sp_id+1*nsp] * Iy); if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                                double fjz = aj2 * (giz - rjri[sp_id+2*nsp] * Iz); if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                                double gkx = gx[addrx+k_1];
                                double gky = gx[addry+k_1];
                                double gkz = gx[addrz+k_1];
                                double fkx = ak2 * gkx; if (kx > 0) { fkx -= kx * gx[addrx-k_1]; } v_kx += fkx * prod_yz;
                                double fky = ak2 * gky; if (ky > 0) { fky -= ky * gx[addry-k_1]; } v_ky += fky * prod_xz;
                                double fkz = ak2 * gkz; if (kz > 0) { fkz -= kz * gx[addrz-k_1]; } v_kz += fkz * prod_xy;
                            }
                        }
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                double *reduce = shared_memory + nsp_per_block * 3 + thread_id;
                __syncthreads();
                reduce[0*THREADS] = v_kx * 2;
                reduce[1*THREADS] = v_ky * 2;
                reduce[2*THREADS] = v_kz * 2;
                for (int i = gout_stride/2; i > 0; i >>= 1) {
                    __syncthreads();
                    if (gout_id < i && pair_ij < shl_pair1 && kidx < ksh1) {
#pragma unroll
                        for (int m = 0; m < 3; ++m) {
                            reduce[m*THREADS] += reduce[m*THREADS+i*nst_per_block];
                        }
                    }
                }
                if (gout_id == 0 && pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, reduce[0*THREADS]);
                    atomicAdd(ejk_aux+ka*3+1, reduce[1*THREADS]);
                    atomicAdd(ejk_aux+ka*3+2, reduce[2*THREADS]);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        double *reduce = shared_memory + nsp_per_block * 3 + thread_id;
        __syncthreads();
        reduce[0*THREADS] = v_ix * 2;
        reduce[1*THREADS] = v_iy * 2;
        reduce[2*THREADS] = v_iz * 2;
        reduce[3*THREADS] = v_jx * 2;
        reduce[4*THREADS] = v_jy * 2;
        reduce[5*THREADS] = v_jz * 2;
        for (int i = gout_stride/2; i > 0; i >>= 1) {
            __syncthreads();
            if (gout_id < i && pair_ij < shl_pair1) {
#pragma unroll
                for (int m = 0; m < 6; ++m) {
                    reduce[m*THREADS] += reduce[m*THREADS+i*nst_per_block];
                }
            }
        }
        if (gout_id == 0 && pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, reduce[0*THREADS]);
            atomicAdd(ejk+ia*3+1, reduce[1*THREADS]);
            atomicAdd(ejk+ia*3+2, reduce[2*THREADS]);
            atomicAdd(ejk+ja*3+0, reduce[3*THREADS]);
            atomicAdd(ejk+ja*3+1, reduce[4*THREADS]);
            atomicAdd(ejk+ja*3+2, reduce[5*THREADS]);
        }
    }
}

extern "C" {
// For exchange energy (density_auxvec==NULL), n_dm must be 1
int ejk_int3c2e_ip1(double *ejk, double *ejk_aux,
                    double *dm, double *density_auxvec, int n_dm,
                    RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int aux_offset, int npairs, int naux)
{
#ifdef USE_SYCL
    sycl::range<2> threads(1, THREADS);
    sycl::range<2> blocks(nbatches_ksh, nbatches_shl_pair);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class ejk_int3c2e_ip1_kernel_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
        ejk_int3c2e_ip1_kernel(
            ejk, ejk_aux, dm, density_auxvec, n_dm, dev_envs,
            shl_pair_offsets, bas_ij_idx, ksh_offsets, gout_stride_lookup,
            ao_pair_loc, aux_offset, npairs, naux,
            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
#else
    cudaFuncSetAttribute(ejk_int3c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    ejk_int3c2e_ip1_kernel<<<blocks, THREADS, shm_size>>>(
            ejk, ejk_aux, dm, density_auxvec, n_dm, *envs,
            shl_pair_offsets, bas_ij_idx, ksh_offsets, gout_stride_lookup,
            ao_pair_loc, aux_offset, npairs, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip1: %s\n", cudaGetErrorString(err));
        return 1;
    }
#endif
    return 0;
}
}
