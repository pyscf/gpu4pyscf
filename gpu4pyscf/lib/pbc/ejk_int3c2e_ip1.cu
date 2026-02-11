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
#include "int3c2e_create_tasks.cuh"

#define LMAX            4
#define LMAX1           (LMAX+1)
#define POOL_SIZE       262144
#define GOUT_WIDTH      54

__global__ static
void ejk_int3c2e_ip1_kernel(double *ejk, double *dm, double *density_auxvec,
                            PBCIntEnvVars envs, uint32_t *pool,
                            uint32_t *bas_ij_idx, int *ksh_offsets, int *ksh_idx,
                            int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int aux_offset, int bvk_naux,
                            float *diffuse_exps, float *diffuse_coefs,
                            float *atom_aux_exps, float log_cutoff
                            #ifdef USE_SYCL
                            , sycl::nd_item<2> &item, char *shm_mem
                            #endif
                            )
{
    #ifdef USE_SYCL
    int ksh_block_id = item.get_group(0);
    int pair_ij = item.get_group(1);
    int thread_id = item.get_local_id(1);

    auto thread_block = item.get_group();
    int &cell0_ksh0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kidx0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kidx1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nksh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &aux_start = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lk = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nroots = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nf = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &kprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nst_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &aux_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nimgs_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nao = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &g_size = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ci = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &cj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double &xi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &yi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &zi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &xjxi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &yjyi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &zjzi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    double &fac = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(thread_block);
    int &img_counts = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);

    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    #else
    int ksh_block_id = blockIdx.y;
    int pair_ij = blockIdx.x;
    int thread_id = threadIdx.x;

    __shared__ int cell0_ksh0, kidx0, kidx1, nksh, aux_start;
    __shared__ int ish, jsh, li, lj, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nimgs_per_block;
    __shared__ int nao;
    __shared__ int g_size;

    __shared__ int expi, expj;
    __shared__ int ci, cj;
    __shared__ double xi, yi, zi;
    __shared__ double xjxi, yjyi, zjzi;
    __shared__ double fac;
    __shared__ int img_counts;

    extern __shared__ double shared_memory[];
    #endif

    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
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
        nksh = kidx1 - kidx0;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ksh = ksh_idx[kidx0];
        ish = bas_ij / bvk_nbas;
        jsh = bas_ij - bvk_nbas * ish;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int lij = li + lj + 1;
        nroots = ((lij + lk) / 2 + 1) * 2;
        iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        nao = ao_loc[envs.cell0_nbas];
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        aux_start = (envs.ao_loc[bvk_nbas+cell0_ksh0] -
                     envs.ao_loc[bvk_nbas] - aux_offset) * ncells;
        int stride_j = li + 2;
        int stride_k = stride_j * (lj + 1);
        g_size = stride_k * (lk + 2);
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

    int ish_cell0 = ish;
    int jsh_cell0 = jsh % envs.nbas;
    if (thread_id == 0) {
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        xi = ri[0];
        yi = ri[1];
        zi = ri[2];
        xjxi = rj[0] - xi;
        yjyi = rj[1] - yi;
        zjzi = rj[2] - zi;
        fac = PI_FAC;
        if (ish_cell0 == jsh_cell0) {
            fac *= .5;
        } else if (ish_cell0 < jsh_cell0) {
            fac = 0;
        }
    }

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int gx_len = g_size * nst_per_block;
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    double v_ix = 0;
    double v_iy = 0;
    double v_iz = 0;
    double v_jx = 0;
    double v_jy = 0;
    double v_jz = 0;

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
                          atom_aux_exps, log_cutoff
                          #ifdef USE_SYCL
                          , item, shm_mem
                          #endif
                          );
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
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xk = env[rk+0];
        double yk = env[rk+1];
        double zk = env[rk+2];
        double dm_tensor[GOUT_WIDTH];
        if (kidx < kidx1) {
            if (density_auxvec == NULL) {
                float div_nfi = c_div_nf[li];
                float div_nfj = c_div_nf[lj];
                float div_nfij = div_nfi * div_nfj;
                size_t pair_offset = ao_pair_loc[pair_ij];
                double *dm_local = dm + pair_offset * bvk_naux + aux_start + kidx - kidx0;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    uint32_t ijk = n*gout_stride+gout_id;
                    if (ijk >= nf) break;
                    uint32_t k = ijk * div_nfij;
                    uint32_t ij = ijk - k * nfi*nfj;
                    dm_tensor[n] = dm_local[ij*bvk_naux + k*nksh];
                }
            } else {
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int ksh_cell0 = (kidx - kidx0) / ncells + cell0_ksh0;
                int k0 = ao_loc[ksh_cell0+bvk_nbas] - ao_loc[bvk_nbas];
                float div_nfi = c_div_nf[li];
                float div_nfj = c_div_nf[lj];
                double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    uint32_t ijk = n*gout_stride+gout_id;
                    if (ijk >= nf) break;
                    uint32_t jk = ijk * div_nfi;
                    uint32_t i = ijk - jk * nfi;
                    uint32_t k = jk * div_nfj;
                    uint32_t j = jk - k * nfj;
                    dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                }
            }
        }

        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        for (int img = img_id; img < img_counts+img_id; img += nimgs_per_block) {
            int img_jk = 0;
            if (img < img_counts) {
                img_jk = img_pool[img];
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double cicj = fac * env[ci+ip] * env[cj+jp];
                if (img >= img_counts || kidx >= kidx1) {
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
                    double xpq = xij - xk - img_coords[kL*3+0];
                    double ypq = yij - yk - img_coords[kL*3+1];
                    double zpq = zij - zk - img_coords[kL*3+2];
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
                    double ak = env[expk+kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                    }
                    rys_roots_rs(nroots, theta, Rpq[3*nst_per_block], omega,
                                 rw, nst_per_block, gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int nst = nst_per_block;
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 1);
                        int gsize = g_size;
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nst_per_block];
                        }
                        int lij = li + lj + 1;
                        double rt = rw[ irys*2   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double s0x, s1x, s2x;
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
                            double Rpa = rjri[n*nst_per_block] * aj_aij;
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
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
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
                            if (img < img_counts && kidx < kidx1) {
                                int lk3 = (lk+2)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix*nst];
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
                        if (img < img_counts && kidx < kidx1) {
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            int i_1 =          nst;
                            int j_1 = stride_j*nst;
                            int k_1 = stride_k*nst;
                            double ai2 = ai * 2;
                            double aj2 = aj * 2;
                            double ak2 = ak * 2;
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
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                                int addry = (iy + jy*stride_j + ky*stride_k + gsize) * nst;
                                int addrz = (iz + jz*stride_j + kz*stride_k + gsize*2) * nst;
                                double Ix = gx[addrx];
                                double Iy = gx[addry];
                                double Iz = gx[addrz];
                                double prod_xy = Ix * Iy * dm_tensor[n];
                                double prod_xz = Ix * Iz * dm_tensor[n];
                                double prod_yz = Iy * Iz * dm_tensor[n];
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
                                double fjx = aj2 * (gix - rjri[0*nst] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                                double fjy = aj2 * (giy - rjri[1*nst] * Iy); if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                                double fjz = aj2 * (giz - rjri[2*nst] * Iz); if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                            }
                        }
                    }
                }
            }
        }

        int ka = bas[ksh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        double *reduce = shared_memory + thread_id;
        __syncthreads();
        reduce[0*THREADS] = v_kx * 2;
        reduce[1*THREADS] = v_ky * 2;
        reduce[2*THREADS] = v_kz * 2;
        int comb_id = gout_id * nimgs_per_block + img_id;
        int comb_stride = gout_stride * nimgs_per_block;
        for (int i = comb_stride/2; i > 0; i >>= 1) {
            __syncthreads();
            if (comb_id < i) {
                reduce[0*THREADS] += reduce[0*THREADS+i*aux_per_block];
                reduce[1*THREADS] += reduce[1*THREADS+i*aux_per_block];
                reduce[2*THREADS] += reduce[2*THREADS+i*aux_per_block];
            }
        }
        if (comb_id == 0 && kidx < kidx1) {
            atomicAdd(ejk+ka*3+0, reduce[0*THREADS]);
            atomicAdd(ejk+ka*3+1, reduce[1*THREADS]);
            atomicAdd(ejk+ka*3+2, reduce[2*THREADS]);
        }
    }
    int ia = bas[ish*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
    int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
    atomicAdd(ejk+ia*3+0, v_ix * 2);
    atomicAdd(ejk+ia*3+1, v_iy * 2);
    atomicAdd(ejk+ia*3+2, v_iz * 2);
    atomicAdd(ejk+ja*3+0, v_jx * 2);
    atomicAdd(ejk+ja*3+1, v_jy * 2);
    atomicAdd(ejk+ja*3+2, v_jz * 2);
}

extern "C" {
int PBCsr_ejk_int3c2e_ip1(double *ejk, double *dm, double *density_auxvec,
                          PBCIntEnvVars *envs, uint32_t *pool,
                          int shm_size, int npairs, int nbatches_ksh,
                          uint32_t *bas_ij_idx, int *ksh_offsets, int *ksh_idx,
                          int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                          int *ao_pair_loc, int aux_offset, int bvk_naux,
                          float *diffuse_exps, float *diffuse_coefs,
                          float *atom_aux_exps, float log_cutoff)
{
    #ifdef USE_SYCL
    sycl::range<2> threads(1, THREADS);
    sycl::range<2> blocks(nbatches_ksh, npairs);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class ejk_int3c2e_ip1_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
        ejk_int3c2e_ip1_kernel(
            ejk, dm, density_auxvec, dev_envs, pool, bas_ij_idx, ksh_offsets, ksh_idx,
            img_idx, img_offsets, gout_stride_lookup, ao_pair_loc, aux_offset, bvk_naux,
            diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff,
            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaFuncSetAttribute(ejk_int3c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(npairs, nbatches_ksh);
    ejk_int3c2e_ip1_kernel<<<blocks, THREADS, shm_size>>>(
            ejk, dm, density_auxvec, *envs, pool, bas_ij_idx, ksh_offsets, ksh_idx,
            img_idx, img_offsets, gout_stride_lookup, ao_pair_loc, aux_offset, bvk_naux,
            diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
