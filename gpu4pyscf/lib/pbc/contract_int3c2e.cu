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
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "pbc.cuh"
#include "int3c2e.cuh"
#include "int3c2e_create_tasks.cuh"

#define LMAX            4
#define LMAX1           (LMAX+1)
#define NF_AUX_MAX      28
#define GOUT_WIDTH      30
#define POOL_SIZE       262144

// lattice sum over j and k for (ij|k)
__global__ static
void contract_int3c2e_dm_kernel(double *out, double *dm, PBCIntEnvVars envs, uint32_t *pool,
                                int *shl_pair_offsets, uint32_t *bas_ij_idx,
                                int *ksh_offsets, int *ksh_idx,
                                int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                                float *diffuse_exps, float *diffuse_coefs,
                                float *atom_aux_exps, float log_cutoff
                                #ifdef USE_SYCL
                                , sycl::nd_item<2> &item, char *shm_mem
                                #endif
                                )
{
    #ifdef USE_SYCL
    int sp_block_id = item.get_group(1);
    int ksh_block_id = item.get_group(0);
    int thread_id = item.get_local_id(1);

    auto thread_block = item.get_group();
    int &shl_pair0       = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &shl_pair1       = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kidx0           = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kidx1           = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li              = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj              = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lk              = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots          = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim           = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim           = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kprim           = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nao             = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride     = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nst_per_block   = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &aux_per_block   = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nimgs_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &img_counts      = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);

    int &ish             = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh             = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double *&expi        = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double *&expj        = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double *&ci          = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double *&cj          = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double &xi           = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &yi           = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &zi           = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &xjxi         = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &yjyi         = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &zjzi         = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &fac          = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);

    double *shared_memory = reinterpret_cast<double *>(shm_mem);
    #else
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int thread_id = threadIdx.x;

    __shared__ int shl_pair0, shl_pair1, kidx0, kidx1;
    __shared__ int li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nao;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nimgs_per_block;
    __shared__ int img_counts;

    __shared__ int ish, jsh;
    __shared__ double *expi, *expj, *ci, *cj;
    __shared__ double xi, yi, zi, xjxi, yjyi, zjzi;
    __shared__ double fac;
    extern __shared__ double shared_memory[];
    #endif
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    double omega = env[PTR_RANGE_OMEGA];
    int nimgs = envs.nimgs;
    if (thread_id == 0) {
        int cell0_ksh0 = ksh_offsets[ksh_block_id];
        int cell0_ksh1 = ksh_offsets[ksh_block_id+1];
        kidx0 = cell0_ksh0 * ncells;
        kidx1 = cell0_ksh1 * ncells;
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        int bas_ij0 = bas_ij_idx[shl_pair0];
        int ish = bas_ij0 / bvk_nbas;
        int jsh = bas_ij0 % bvk_nbas;
        int ksh = ksh_idx[kidx0];
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int lij = li + lj;
        nroots = ((lij + lk) / 2 + 1) * 2;
        iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        nao = ao_loc[envs.nbas];
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
        aux_per_block = min(nst_per_block, WARP_SIZE);
        nimgs_per_block = nst_per_block / aux_per_block;
    }
    __syncthreads();
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;
    int img_id = st_id / aux_per_block;
    int aux_id = st_id - img_id * aux_per_block;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfk = c_nf[lk];
    int nfij = nfi * nfj;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nst_per_block;
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int *idx_i = (int*)(shared_memory + nst_per_block*(g_size*3+nroots*2+7));
    int *idx_j = idx_i + nfi * 3;
    int *idx_k = idx_j + nfj * 3;
    if (thread_id < nfi * 3) {
        idx_i[thread_id] = lex_xyz_address(li, thread_id) * nst_per_block;
        idx_i[thread_id] += (thread_id % 3) * gx_len;
    }
    if (thread_id < nfj * 3) {
        idx_j[thread_id] = lex_xyz_address(lj, thread_id) * stride_j * nst_per_block;
    }
    if (thread_id < nfk * 3) {
        idx_k[thread_id] = lex_xyz_address(lk, thread_id) * stride_k * nst_per_block;
    }

    uint32_t *img_pool = pool + get_smid() * POOL_SIZE;
    for (int kidx = kidx0+aux_id; kidx < kidx1+aux_id; kidx += aux_per_block) {
        double vj_xyz[NF_AUX_MAX];
        for (int n = 0; n < NF_AUX_MAX; ++n) {
            vj_xyz[n] = 0;
        }
        int ksh;
        if (kidx < kidx1) {
            ksh = ksh_idx[kidx];
        } else {
            ksh = ksh_idx[kidx0];
        }
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        for (int pair_ij = shl_pair0; pair_ij < shl_pair1; pair_ij++) {
            __syncthreads();
            uint32_t bas_ij = bas_ij_idx[pair_ij];
            if (thread_id == 0) {
                ish = bas_ij / bvk_nbas;
                jsh = bas_ij % bvk_nbas;
                expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                xi = ri[0];
                yi = ri[1];
                zi = ri[2];
                xjxi = rj[0] - xi;
                yjyi = rj[1] - yi;
                zjzi = rj[2] - zi;
                int ish_cell0 = ish;
                int jsh_cell0 = jsh % envs.nbas;
                fac = PI_FAC;
                if (ish_cell0 < jsh_cell0 || kidx >= kidx1) {
                    fac = 0;
                } else if (ish_cell0 == jsh_cell0) {
                    fac *= .5;
                }
            }
            __syncthreads();
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jsh];
            double *dm_local = dm + j0 * nao + i0;
            if (thread_id == 0) {
                img_counts = 0;
            }
            __syncthreads();
            int kidx0p = kidx - aux_id;
            _filter_jk_images(img_counts, img_pool, envs, pair_ij, bas_ij,
                              kidx0p, min(kidx0p+aux_per_block, kidx1), li, lj,
                              ksh_idx, img_idx, img_offsets, diffuse_exps, diffuse_coefs,
                              atom_aux_exps, log_cutoff);
            __syncthreads();
            if (img_counts == 0) {
                continue;
            }
            for (int img = img_id; img < img_counts+img_id; img += nimgs_per_block) {
                int img_jk = 0;
                if (img < img_counts) {
                    img_jk = img_pool[img];
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp - jprim * ip;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double theta_ij = ai * aj_aij;
                    double cicj = fac * ci[ip] * cj[jp];
                    if (img >= img_counts) {
                        cicj = 0;
                    }
                    if (gout_id == 0) {
                        int jL = img_jk / nimgs;
                        int kL = img_jk - nimgs * jL;
                        double xjLxi = xjxi + img_coords[jL*3+0];
                        double yjLyi = yjyi + img_coords[jL*3+1];
                        double zjLzi = zjzi + img_coords[jL*3+2];
                        double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                        double Kab = theta_ij * rr_ij;
                        double fac_ij = exp(-Kab);
                        double xij = xjLxi * aj_aij + xi;
                        double yij = yjLyi * aj_aij + yi;
                        double zij = zjLzi * aj_aij + zi;
                        double xpq = xij - rk[0] - img_coords[kL*3+0];
                        double ypq = yij - rk[1] - img_coords[kL*3+1];
                        double zpq = zij - rk[2] - img_coords[kL*3+2];
                        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                        rjri[0*nst_per_block] = xjLxi;
                        rjri[1*nst_per_block] = yjLyi;
                        rjri[2*nst_per_block] = zjLzi;
                        Rpq[0*nst_per_block] = xpq;
                        Rpq[1*nst_per_block] = ypq;
                        Rpq[2*nst_per_block] = zpq;
                        Rpq[3*nst_per_block] = rr;
                        gx[gx_len] = cicj * fac_ij;
                    }
                    for (int kp = 0; kp < kprim; ++kp) {
                        double ak = expk[kp];
                        double theta = aij * ak / (aij + ak);
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[0] = ck[kp] / (aij*ak*sqrt(aij+ak));
                        }
                        rys_roots_rs(nroots, theta, Rpq[3*nst_per_block], omega,
                                     rw, nst_per_block, gout_id, gout_stride);
                        double s0x, s1x, s2x;
                        for (int irys = 0; irys < nroots; ++irys) {
                            __syncthreads();
                            if (gout_id == 0) {
                                gx[gx_len*2] = rw[(irys*2+1)*nst_per_block];
                            }
                            double rt = rw[ irys*2   *nst_per_block];
                            double rt_aa = rt / (aij + ak);
                            int lij = li + lj;
                            if (lij > 0) {
                                __syncthreads();
                                double rt_aij = rt_aa * ak;
                                double b10 = .5/aij * (1 - rt_aij);
                                // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                                for (int n = gout_id; n < 3; n += gout_stride) {
                                    double *_gx = gx + n * gx_len;
                                    double xpa = rjri[n*nst_per_block] * aj_aij;
                                    //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                                    double c0x = xpa - rt_aij * Rpq[n*nst_per_block];
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
                            }

                            if (lk > 0) {
                                int lij3 = (lij+1)*3;
                                double rt_ak  = rt_aa * aij;
                                double b00 = .5 * rt_aa;
                                double b01 = .5/ak  * (1 - rt_ak);
                                for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                                    __syncthreads();
                                    int i = n / 3; //for i in range(lij+1):
                                    int _ix = n % 3; // TODO: remove _ix for nroots > 2
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
                                    for (int k = 1; k < lk; ++k) {
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
                            }

                            if (lj > 0) {
                                __syncthreads();
                                if (img < img_counts && kidx < kidx1) {
                                    int lk3 = (lk+1)*3;
                                    for (int m = gout_id; m < lk3; m += gout_stride) {
                                        int k = m / 3;
                                        int _ix = m % 3;
                                        double xjxi = rjri[_ix*nst_per_block];
                                        double *_gx = gx + (_ix*g_size + k*stride_k)
                                            * nst_per_block;
                                        for (int j = 0; j < lj; ++j) {
                                            int ij = lij + j*li; // = (lij-j) + j*stride_j;
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
                            if (img < img_counts && kidx < kidx1) {
                                float div_nfi = c_div_nf[li];
                                for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                                    uint32_t j = ij * div_nfi;
                                    uint32_t i = ij - nfi * j;
                                    double dm_ij = dm_local[j*nao+i];
                                    int ij_addrx = idx_i[i*3+0] + idx_j[j*3+0];
                                    int ij_addry = idx_i[i*3+1] + idx_j[j*3+1];
                                    int ij_addrz = idx_i[i*3+2] + idx_j[j*3+2];
#pragma unroll
                                    for (int k = 0; k < NF_AUX_MAX; ++k) {
                                        if (k >= nfk) break;
                                        int addrx = ij_addrx + idx_k[k*3+0];
                                        int addry = ij_addry + idx_k[k*3+1];
                                        int addrz = ij_addrz + idx_k[k*3+2];
                                        vj_xyz[k] += gx[addrx] * gx[addry] * gx[addrz] * dm_ij;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (nimgs_per_block > 1) {
            double *reduce = shared_memory + thread_id;
#pragma unroll
            for (int n = 0; n < NF_AUX_MAX; ++n) {
                if (n >= nfk) break;
                __syncthreads();
                reduce[0] = vj_xyz[n];
                for (int i = nimgs_per_block/2; i > 0; i >>= 1) {
                    __syncthreads();
                    if (img_id < i) {
                        reduce[0] += reduce[i*aux_per_block];
                    }
                }
                if (img_id == 0) {
                    vj_xyz[n] = reduce[0];
                }
            }
        }
        if (img_id == 0 && kidx < kidx1) {
            int cell0_ksh0 = kidx0 / ncells;
            int nksh = kidx1 - kidx0;
            int aux_start = (envs.ao_loc[bvk_nbas+cell0_ksh0] -
                             envs.ao_loc[bvk_nbas]) * ncells;
            double *vj = out + aux_start + kidx - kidx0;
#pragma unroll
            for (int k = 0; k < NF_AUX_MAX; ++k) {
                if (k >= nfk) break;
                atomicAdd(vj + k*nksh, vj_xyz[k]);
            }
        }
    }
}

__global__ static
void contract_int3c2e_auxvec_kernel(double *out, double *auxvec,
                                    PBCIntEnvVars envs, uint32_t *pool,
                                    uint32_t *bas_ij_idx, int *ksh_offsets, int *ksh_idx,
                                    int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                                    float *diffuse_exps, float *diffuse_coefs,
                                    float *atom_aux_exps, float log_cutoff
                                    #ifdef USE_SYCL
                                    , sycl::nd_item<2> &item, char *shm_mem
                                    #endif
                                    )
{
    #ifdef USE_SYCL
    int ksh_block_id = item.get_group(0);
    int thread_id = item.get_local_id(1);
    int pair_ij = item.get_group(1);

    auto thread_block = item.get_group();
    int &cell0_ksh0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kidx0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kidx1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nst_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &aux_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nimgs_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double *&expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double *&expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double *&ci = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double *&cj = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double &xi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &yi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &zi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &xjxi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &yjyi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &zjzi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &fac = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    int &img_counts = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int ksh_block_id = blockIdx.y;
    int thread_id = threadIdx.x;
    int pair_ij = blockIdx.x;

    __shared__ int cell0_ksh0, kidx0, kidx1;
    __shared__ int ish, jsh, li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nimgs_per_block;
    __shared__ double *expi, *expj, *ci, *cj;
    __shared__ double xi, yi, zi, xjxi, yjyi, zjzi;
    __shared__ double fac;
    __shared__ int img_counts;

    extern __shared__ double shared_memory[];
    #endif

    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    double omega = env[PTR_RANGE_OMEGA];
    int nimgs = envs.nimgs;
    if (thread_id == 0) {
        cell0_ksh0 = ksh_offsets[ksh_block_id];
        int cell0_ksh1 = ksh_offsets[ksh_block_id+1];
        kidx0 = cell0_ksh0 * ncells;
        kidx1 = cell0_ksh1 * ncells;
        ish = bas_ij / bvk_nbas;
        jsh = bas_ij % bvk_nbas;
        int ksh = ksh_idx[kidx0];
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int lij = li + lj;
        nroots = ((lij + lk) / 2 + 1) * 2;
        iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        xi = ri[0];
        yi = ri[1];
        zi = ri[2];
        xjxi = rj[0] - xi;
        yjyi = rj[1] - yi;
        zjzi = rj[2] - zi;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.nbas;
        fac = PI_FAC;
        if (ish_cell0 < jsh_cell0) {
            fac = 0;
        } else if (ish_cell0 == jsh_cell0) {
            fac *= .5;
        }
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
        aux_per_block = min(nst_per_block, WARP_SIZE);
        nimgs_per_block = nst_per_block / aux_per_block;
    }
    __syncthreads();
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;
    int img_id = st_id / aux_per_block;
    int aux_id = st_id - img_id * aux_per_block;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfk = c_nf[lk];
    int nfij = nfi * nfj;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nst_per_block;
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int *idx_i = (int*)(shared_memory + nst_per_block*(g_size*3+nroots*2+7));
    int *idx_j = idx_i + nfi * 3;
    int *idx_k = idx_j + nfj * 3;
    if (thread_id < nfi * 3) {
        idx_i[thread_id] = lex_xyz_address(li, thread_id) * nst_per_block;
        idx_i[thread_id] += (thread_id % 3) * gx_len;
    }
    if (thread_id < nfj * 3) {
        idx_j[thread_id] = lex_xyz_address(lj, thread_id) * stride_j * nst_per_block;
    }
    if (thread_id < nfk * 3) {
        idx_k[thread_id] = lex_xyz_address(lk, thread_id) * stride_k * nst_per_block;
    }

    double vj[GOUT_WIDTH];
    for (int n = 0; n < GOUT_WIDTH; ++n) {
        vj[n] = 0;
    }

    uint32_t *img_pool = pool + get_smid() * POOL_SIZE;
    for (int kidx = kidx0+aux_id; kidx < kidx1+aux_id; kidx += aux_per_block) {
        __syncthreads();
        if (thread_id == 0) {
            img_counts = 0;
        }
        __syncthreads();
        int kidx0p = kidx - aux_id;
        _filter_jk_images(img_counts, img_pool, envs, pair_ij, bas_ij,
                          kidx0p, min(kidx0p+aux_per_block, kidx1), li, lj,
                          ksh_idx, img_idx, img_offsets, diffuse_exps, diffuse_coefs,
                          atom_aux_exps, log_cutoff);
        __syncthreads();
        if (img_counts == 0) {
            continue;
        }

        int ksh;
        if (kidx < kidx1) {
            ksh = ksh_idx[kidx];
        } else {
            ksh = ksh_idx[kidx0];
        }
        int ksh_cell0 = (kidx - kidx0) / ncells + cell0_ksh0;
        int k0 = ao_loc[ksh_cell0+bvk_nbas] - ao_loc[bvk_nbas];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        for (int img = img_id; img < img_counts+img_id; img += nimgs_per_block) {
            int img_jk = 0;
            if (img < img_counts) {
                img_jk = img_pool[img];
            }
            int jL = img_jk / nimgs;
            int kL = img_jk - nimgs * jL;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double cicj = fac * ci[ip] * cj[jp];
                if (img >= img_counts || kidx >= kidx1) {
                    cicj = 0;
                }
                if (gout_id == 0) {
                    double xjLxi = xjxi + img_coords[jL*3+0];
                    double yjLyi = yjyi + img_coords[jL*3+1];
                    double zjLzi = zjzi + img_coords[jL*3+2];
                    double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                    double Kab = theta_ij * rr_ij;
                    double fac_ij = exp(-Kab);
                    double xij = xjLxi * aj_aij + xi;
                    double yij = yjLyi * aj_aij + yi;
                    double zij = zjLzi * aj_aij + zi;
                    double xpq = xij - rk[0] - img_coords[kL*3+0];
                    double ypq = yij - rk[1] - img_coords[kL*3+1];
                    double zpq = zij - rk[2] - img_coords[kL*3+2];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    rjri[0*nst_per_block] = xjLxi;
                    rjri[1*nst_per_block] = yjLyi;
                    rjri[2*nst_per_block] = zjLzi;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                    Rpq[3*nst_per_block] = rr;
                    gx[gx_len] = cicj * fac_ij;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = expk[kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = ck[kp] / (aij*ak*sqrt(aij+ak));
                    }
                    rys_roots_rs(nroots, theta, Rpq[3*nst_per_block], omega,
                                 rw, nst_per_block, gout_id, gout_stride);
                    double s0x, s1x, s2x;
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nst_per_block];
                        }
                        double rt = rw[ irys*2   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        int lij = li + lj;
                        if (lij > 0) {
                            __syncthreads();
                            double rt_aij = rt_aa * ak;
                            double b10 = .5/aij * (1 - rt_aij);
                            // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                            for (int n = gout_id; n < 3; n += gout_stride) {
                                double *_gx = gx + n * gx_len;
                                double xpa = rjri[n*nst_per_block] * aj_aij;
                                //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                                double c0x = xpa - rt_aij * Rpq[n*nst_per_block];
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
                        }

                        if (lk > 0) {
                            int lij3 = (lij+1)*3;
                            double rt_ak  = rt_aa * aij;
                            double b00 = .5 * rt_aa;
                            double b01 = .5/ak  * (1 - rt_ak );
                            for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                                __syncthreads();
                                int i = n / 3; //for i in range(lij+1):
                                int _ix = n % 3; // TODO: remove _ix for nroots > 2
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
                                for (int k = 1; k < lk; ++k) {
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
                        }

                        if (lj > 0) {
                            __syncthreads();
                            if (img < img_counts && kidx < kidx1) {
                                int lk3 = (lk+1)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix*nst_per_block];
                                    double *_gx = gx + (_ix*g_size + k*stride_k) * nst_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = lij + j*li; // = (lij-j) + j*stride_j;
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
                        if (img < img_counts && kidx < kidx1) {
                            float div_nfi = c_div_nf[li];
                            for (int k = 0; k < nfk; ++k) {
                                int kx = idx_k[k*3+0];
                                int ky = idx_k[k*3+1];
                                int kz = idx_k[k*3+2];
                                double rho = auxvec[k0+k];
#pragma unroll
                                for (int n = 0; n < GOUT_WIDTH; n++) {
                                    uint32_t ij = gout_id + n * gout_stride;
                                    if (ij >= nfij) break;
                                    uint32_t j = ij * div_nfi;
                                    uint32_t i = ij - nfi * j;
                                    int addrx = idx_i[i*3+0] + idx_j[j*3+0] + kx;
                                    int addry = idx_i[i*3+1] + idx_j[j*3+1] + ky;
                                    int addrz = idx_i[i*3+2] + idx_j[j*3+2] + kz;
                                    vj[n] += gx[addrx] * gx[addry] * gx[addrz] * rho;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    int nao = ao_loc[bvk_nbas];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    double *vj_ij = out + i0 * nao + j0;
#pragma unroll
    for (int n = 0; n < GOUT_WIDTH; n++) {
        int ij = n*gout_stride+gout_id;
        if (ij >= nfij) break;
        int i = ij % nfi;
        int j = ij / nfi;
        atomicAdd(vj_ij + i*nao+j, vj[n]);
    }
}

extern "C" {
int PBCcontract_int3c2e_dm(double *out, double *dm,
                           PBCIntEnvVars *envs, uint32_t *pool, int shm_size,
                           int nbatches_shl_pair, int nbatches_ksh, int *shl_pair_offsets,
                           uint32_t *bas_ij_idx, int *ksh_offsets, int *ksh_idx,
                           int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                           float *diffuse_exps, float *diffuse_coefs,
                           float *atom_aux_exps, float log_cutoff)
{
    #ifdef USE_SYCL
    sycL::range<2> blocks(nbatches_ksh, nbatches_shl_pair);
    sycl::range<2> threads(1, THREADS);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class contract_int3c2e_dm_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
        contract_int3c2e_dm_kernel(out, dm, dev_envs, pool, shl_pair_offsets, bas_ij_idx, ksh_offsets, ksh_idx,
                                   img_idx, img_offsets, gout_stride_lookup,
                                   diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff,
                                   item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaFuncSetAttribute(contract_int3c2e_dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    contract_int3c2e_dm_kernel<<<blocks, THREADS, shm_size>>>(
            out, dm, *envs, pool, shl_pair_offsets, bas_ij_idx, ksh_offsets, ksh_idx,
            img_idx, img_offsets, gout_stride_lookup,
            diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int PBCcontract_int3c2e_auxvec(double *out, double *auxvec,
                               PBCIntEnvVars *envs, uint32_t *pool, int shm_size,
                               int npairs, int nbatches_ksh, uint32_t *bas_ij_idx,
                               int *ksh_offsets, int *ksh_idx,
                               int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                               float *diffuse_exps, float *diffuse_coefs,
                               float *atom_aux_exps, float log_cutoff)
{
    #ifdef USE_SYCL
    sycl::range<2> blocks(nbatches_ksh, npairs);
    sycl::range<2> threads(1, THREADS);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class contract_int3c2e_auxvec_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
    contract_int3c2e_auxvec_kernel(
            out, auxvec, dev_envs, pool, bas_ij_idx, ksh_offsets, ksh_idx,
            img_idx, img_offsets, gout_stride_lookup,
            diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff,
            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaFuncSetAttribute(contract_int3c2e_auxvec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(npairs, nbatches_ksh);
    contract_int3c2e_auxvec_kernel<<<blocks, THREADS, shm_size>>>(
            out, auxvec, *envs, pool, bas_ij_idx, ksh_offsets, ksh_idx,
            img_idx, img_offsets, gout_stride_lookup,
            diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
