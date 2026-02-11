/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
#define GOUT_WIDTH      54
#define POOL_SIZE       262144

// lattice sum over j and k for (ij|k)
__global__ static
void pbc_int3c2e_latsum23_kernel(double *out, PBCIntEnvVars envs, uint32_t *pool,
                                 uint32_t *bas_ij_idx, int *ksh_offsets, int *ksh_idx,
                                 int *img_idx, uint32_t *sp_img_offsets,
                                 int *gout_stride_lookup, int *ao_pair_loc,
                                 int ao_pair_offset, int aux_offset, int bvk_naux, int to_sph,
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
    int &img_counts = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);

    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    #else
    int ksh_block_id = blockIdx.y;
    int pair_ij = blockIdx.x;
    int thread_id = threadIdx.x;

    __shared__ int kidx0, kidx1, nksh, aux_start;
    __shared__ int ish, jsh, li, lj, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nimgs_per_block;
    __shared__ double *expi, *expj, *ci, *cj;
    __shared__ double xi, yi, zi, xjxi, yjyi, zjzi;
    __shared__ int img_counts;

    extern __shared__ double shared_memory[];
    #endif

    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    double omega = env[PTR_RANGE_OMEGA];
    int nimgs = envs.nimgs;
    if (thread_id == 0) {
        int cell0_ksh0 = ksh_offsets[ksh_block_id];
        int cell0_ksh1 = ksh_offsets[ksh_block_id+1];
        kidx0 = cell0_ksh0 * ncells;
        kidx1 = cell0_ksh1 * ncells;
        nksh = kidx1 - kidx0;
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
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        aux_start = (envs.ao_loc[bvk_nbas+cell0_ksh0] -
                     envs.ao_loc[bvk_nbas] - aux_offset) * ncells;
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
        __syncthreads();
        if (thread_id == 0) {
            img_counts = 0;
        }
        __syncthreads();
        int kidx0p = kidx - aux_id;
        _filter_jk_images(img_counts, img_pool, envs, pair_ij, bas_ij,
                          kidx0p, min(kidx0p+aux_per_block, kidx1), li, lj,
                          ksh_idx, img_idx, sp_img_offsets, diffuse_exps, diffuse_coefs,
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
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

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
                double cicj = PI_FAC * ci[ip] * cj[jp];
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

                        // hrr
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
                        if (img < img_counts && kidx < kidx1) {
                            float div_nfi = c_div_nf[li];
                            float div_nfk = c_div_nf[lk];
#pragma unroll
                            for (int n = 0; n < GOUT_WIDTH; ++n) {
                                uint32_t ijk = n*gout_stride+gout_id;
                                if (ijk >= nf) break;
                                uint32_t ij = ijk * div_nfk;
                                uint32_t k = ijk - nfk * ij;
                                uint32_t j = ij * div_nfi;
                                uint32_t i = ij - nfi * j;
                                int addrx = idx_i[i*3+0] + idx_j[j*3+0] + idx_k[k*3+0];
                                int addry = idx_i[i*3+1] + idx_j[j*3+1] + idx_k[k*3+1];
                                int addrz = idx_i[i*3+2] + idx_j[j*3+2] + idx_k[k*3+2];
                                gout[n] += gx[addrx] * gx[addry] * gx[addrz];
                            }
                        }
                    }
                }
            }
        }

        if (nimgs_per_block > 1) {
            double *reduce = shared_memory + thread_id;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                if (n*gout_stride >= nf) break;
                __syncthreads();
                reduce[0] = gout[n];
                for (int i = nimgs_per_block/2; i > 0; i >>= 1) {
                    __syncthreads();
                    if (img_id < i) {
                        reduce[0] += reduce[i*aux_per_block];
                    }
                }
                if (img_id == 0) {
                    gout[n] = reduce[0];
                }
            }
        }
        __syncthreads();

        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        double *j3c = out + pair_offset * bvk_naux + aux_start + kidx - kidx0;
        int i_stride = bvk_naux;
        int aux_stride = nksh;
        double *out_local = j3c;
        if (to_sph && (li > 1 || lj > 1)) {
            i_stride = aux_per_block * nfk;
            aux_stride = aux_per_block;
            double *c2s_pool = (double *)img_pool;
            out_local = c2s_pool + aux_id;
        }
        if (img_id == 0 && kidx < kidx1) {
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ijk = n*gout_stride+gout_id;
                if (ijk >= nf) break;
                int ij = ijk / nfk;
                int k  = ijk - nfk * ij;
                out_local[ij*i_stride + k*aux_stride] = gout[n];
            }
        }
        __syncthreads();
        if (kidx < kidx1 && to_sph && (li > 1 || lj > 1)) {
            int di = li * 2 + 1;
            int i_stride = aux_per_block * nfk;
            int j_stride = bvk_naux * di;
            double *c2s_pool = (double *)img_pool;
            double *inp_local = c2s_pool + aux_id;
            // Note each block within the compressed data in the input is transposed
            // for block with shape [nfi,nfj], i is accessed with smaller strides
            int comb_id = gout_id * nimgs_per_block + img_id;
            int comb_stride = nimgs_per_block * gout_stride;
            for (int k = comb_id; k < nfk; k += comb_stride) {
                for (int j = 0; j < nfj; j++) {
                    double *inp = inp_local + (j * nfi * nfk + k) * aux_per_block;
                    for (int i = 0; i < di; i++) {
                        double *sph_out = j3c + i * bvk_naux + k * nksh;
                        double s = 0;
                        // cart2sph for i
                        switch (li*li+i) {
                        case 0: { // l=0, m=0
                            s += inp[i_stride*0] * 1;
                        } break;
                        case 1: { // l=1, m=0
                            s += inp[i_stride*0] * 1;
                        } break;
                        case 2: { // l=1, m=1
                            s += inp[i_stride*1] * 1;
                        } break;
                        case 3: { // l=1, m=2
                            s += inp[i_stride*2] * 1;
                        } break;
                        case 4: { // l=2, m=0
                            s += inp[i_stride*1] * 1.092548430592079070;
                        } break;
                        case 5: { // l=2, m=1
                            s += inp[i_stride*4] * 1.092548430592079070;
                        } break;
                        case 6: { // l=2, m=2
                            s += inp[i_stride*0] * -0.315391565252520002;
                            s += inp[i_stride*3] * -0.315391565252520002;
                            s += inp[i_stride*5] * 0.630783130505040012;
                        } break;
                        case 7: { // l=2, m=3
                            s += inp[i_stride*2] * 1.092548430592079070;
                        } break;
                        case 8: { // l=2, m=4
                            s += inp[i_stride*0] * 0.546274215296039535;
                            s += inp[i_stride*3] * -0.546274215296039535;
                        } break;
                        case 9: { // l=3, m=0
                            s += inp[i_stride*1] * 1.770130769779930531;
                            s += inp[i_stride*6] * -0.590043589926643510;
                        } break;
                        case 10: { // l=3, m=1
                            s += inp[i_stride*4] * 2.890611442640554055;
                        } break;
                        case 11: { // l=3, m=2
                            s += inp[i_stride*1] * -0.457045799464465739;
                            s += inp[i_stride*6] * -0.457045799464465739;
                            s += inp[i_stride*8] * 1.828183197857862944;
                        } break;
                        case 12: { // l=3, m=3
                            s += inp[i_stride*2] * -1.119528997770346170;
                            s += inp[i_stride*7] * -1.119528997770346170;
                            s += inp[i_stride*9] * 0.746352665180230782;
                        } break;
                        case 13: { // l=3, m=4
                            s += inp[i_stride*0] * -0.457045799464465739;
                            s += inp[i_stride*3] * -0.457045799464465739;
                            s += inp[i_stride*5] * 1.828183197857862944;
                        } break;
                        case 14: { // l=3, m=5
                            s += inp[i_stride*2] * 1.445305721320277020;
                            s += inp[i_stride*7] * -1.445305721320277020;
                        } break;
                        case 15: { // l=3, m=6
                            s += inp[i_stride*0] * 0.590043589926643510;
                            s += inp[i_stride*3] * -1.770130769779930530;
                        } break;
                        case 16: { // l=4, m=0
                            s += inp[i_stride*1] * 2.503342941796704538;
                            s += inp[i_stride*6] * -2.503342941796704530;
                        } break;
                        case 17: { // l=4, m=1
                            s += inp[i_stride*4] * 5.310392309339791593;
                            s += inp[i_stride*11] * -1.770130769779930530;
                        } break;
                        case 18: { // l=4, m=2
                            s += inp[i_stride*1] * -0.946174695757560014;
                            s += inp[i_stride*6] * -0.946174695757560014;
                            s += inp[i_stride*8] * 5.677048174545360108;
                        } break;
                        case 19: { // l=4, m=3
                            s += inp[i_stride*4] * -2.007139630671867500;
                            s += inp[i_stride*11] * -2.007139630671867500;
                            s += inp[i_stride*13] * 2.676186174229156671;
                        } break;
                        case 20: { // l=4, m=4
                            s += inp[i_stride*0] * 0.317356640745612911;
                            s += inp[i_stride*3] * 0.634713281491225822;
                            s += inp[i_stride*5] * -2.538853125964903290;
                            s += inp[i_stride*10] * 0.317356640745612911;
                            s += inp[i_stride*12] * -2.538853125964903290;
                            s += inp[i_stride*14] * 0.846284375321634430;
                        } break;
                        case 21: { // l=4, m=5
                            s += inp[i_stride*2] * -2.007139630671867500;
                            s += inp[i_stride*7] * -2.007139630671867500;
                            s += inp[i_stride*9] * 2.676186174229156671;
                        } break;
                        case 22: { // l=4, m=6
                            s += inp[i_stride*0] * -0.473087347878780002;
                            s += inp[i_stride*5] * 2.838524087272680054;
                            s += inp[i_stride*10] * 0.473087347878780009;
                            s += inp[i_stride*12] * -2.838524087272680050;
                        } break;
                        case 23: { // l=4, m=7
                            s += inp[i_stride*2] * 1.770130769779930531;
                            s += inp[i_stride*7] * -5.310392309339791590;
                        } break;
                        case 24: { // l=4, m=8
                            s += inp[i_stride*0] * 0.625835735449176134;
                            s += inp[i_stride*3] * -3.755014412695056800;
                            s += inp[i_stride*10] * 0.625835735449176134;
                        } break;
                        case 25: { // l=5, m=0
                            s += inp[i_stride*1] * 3.281910284200850514;
                            s += inp[i_stride*6] * -6.563820568401701020;
                            s += inp[i_stride*15] * 0.656382056840170102;
                        } break;
                        case 26: { // l=5, m=1
                            s += inp[i_stride*4] * 8.302649259524165115;
                            s += inp[i_stride*11] * -8.302649259524165110;
                        } break;
                        case 27: { // l=5, m=2
                            s += inp[i_stride*1] * -1.467714898305751160;
                            s += inp[i_stride*6] * -0.978476598870500779;
                            s += inp[i_stride*8] * 11.741719186446009300;
                            s += inp[i_stride*15] * 0.489238299435250387;
                            s += inp[i_stride*17] * -3.913906395482003100;
                        } break;
                        case 28: { // l=5, m=3
                            s += inp[i_stride*4] * -4.793536784973323750;
                            s += inp[i_stride*11] * -4.793536784973323750;
                            s += inp[i_stride*13] * 9.587073569946647510;
                        } break;
                        case 29: { // l=5, m=4
                            s += inp[i_stride*1] * 0.452946651195696921;
                            s += inp[i_stride*6] * 0.905893302391393842;
                            s += inp[i_stride*8] * -5.435359814348363050;
                            s += inp[i_stride*15] * 0.452946651195696921;
                            s += inp[i_stride*17] * -5.435359814348363050;
                            s += inp[i_stride*19] * 3.623573209565575370;
                        } break;
                        case 30: { // l=5, m=5
                            s += inp[i_stride*2] * 1.754254836801353946;
                            s += inp[i_stride*7] * 3.508509673602707893;
                            s += inp[i_stride*9] * -4.678012898136943850;
                            s += inp[i_stride*16] * 1.754254836801353946;
                            s += inp[i_stride*18] * -4.678012898136943850;
                            s += inp[i_stride*20] * 0.935602579627388771;
                        } break;
                        case 31: { // l=5, m=6
                            s += inp[i_stride*0] * 0.452946651195696921;
                            s += inp[i_stride*3] * 0.905893302391393842;
                            s += inp[i_stride*5] * -5.435359814348363050;
                            s += inp[i_stride*10] * 0.452946651195696921;
                            s += inp[i_stride*12] * -5.435359814348363050;
                            s += inp[i_stride*14] * 3.623573209565575370;
                        } break;
                        case 32: { // l=5, m=7
                            s += inp[i_stride*2] * -2.396768392486661870;
                            s += inp[i_stride*9] * 4.793536784973323755;
                            s += inp[i_stride*16] * 2.396768392486661877;
                            s += inp[i_stride*18] * -4.793536784973323750;
                        } break;
                        case 33: { // l=5, m=8
                            s += inp[i_stride*0] * -0.489238299435250389;
                            s += inp[i_stride*3] * 0.978476598870500775;
                            s += inp[i_stride*5] * 3.913906395482003101;
                            s += inp[i_stride*10] * 1.467714898305751163;
                            s += inp[i_stride*12] * -11.741719186446009300;
                        } break;
                        case 34: { // l=5, m=9
                            s += inp[i_stride*2] * 2.075662314881041278;
                            s += inp[i_stride*7] * -12.453973889286247600;
                            s += inp[i_stride*16] * 2.075662314881041278;
                        } break;
                        case 35: { // l=5, m=10
                            s += inp[i_stride*0] * 0.656382056840170102;
                            s += inp[i_stride*3] * -6.563820568401701020;
                            s += inp[i_stride*10] * 3.281910284200850514;
                        } break;
                        case 36: { // l=6, m=0
                            s += inp[i_stride*1] * 4.0991046311514863;
                            s += inp[i_stride*6] * -13.6636821038382887;
                            s += inp[i_stride*15] * 4.0991046311514863;
                        } break;
                        case 37: { // l=6, m=1
                            s += inp[i_stride*4] * 11.8330958111587634;
                            s += inp[i_stride*11] * -23.6661916223175268;
                            s += inp[i_stride*22] * 2.3666191622317525;
                        } break;
                        case 38: { // l=6, m=2
                            s += inp[i_stride*1] * -2.0182596029148963;
                            s += inp[i_stride*8] * 20.1825960291489679;
                            s += inp[i_stride*15] * 2.0182596029148963;
                            s += inp[i_stride*17] * -20.1825960291489679;
                        } break;
                        case 39: { // l=6, m=3
                            s += inp[i_stride*4] * -8.2908473356343109;
                            s += inp[i_stride*11] * -5.5272315570895412;
                            s += inp[i_stride*13] * 22.1089262283581647;
                            s += inp[i_stride*22] * 2.7636157785447706;
                            s += inp[i_stride*24] * -7.3696420761193888;
                        } break;
                        case 40: { // l=6, m=4
                            s += inp[i_stride*1] * 0.9212052595149236;
                            s += inp[i_stride*6] * 1.8424105190298472;
                            s += inp[i_stride*8] * -14.7392841522387776;
                            s += inp[i_stride*15] * 0.9212052595149236;
                            s += inp[i_stride*17] * -14.7392841522387776;
                            s += inp[i_stride*19] * 14.7392841522387776;
                        } break;
                        case 41: { // l=6, m=5
                            s += inp[i_stride*4] * 2.9131068125936568;
                            s += inp[i_stride*11] * 5.8262136251873136;
                            s += inp[i_stride*13] * -11.6524272503746271;
                            s += inp[i_stride*22] * 2.9131068125936568;
                            s += inp[i_stride*24] * -11.6524272503746271;
                            s += inp[i_stride*26] * 4.6609709001498505;
                        } break;
                        case 42: { // l=6, m=6
                            s += inp[i_stride*0] * -0.3178460113381421;
                            s += inp[i_stride*3] * -0.9535380340144264;
                            s += inp[i_stride*5] * 5.7212282040865583;
                            s += inp[i_stride*10] * -0.9535380340144264;
                            s += inp[i_stride*12] * 11.4424564081731166;
                            s += inp[i_stride*14] * -7.6283042721154111;
                            s += inp[i_stride*21] * -0.3178460113381421;
                            s += inp[i_stride*23] * 5.7212282040865583;
                            s += inp[i_stride*25] * -7.6283042721154111;
                            s += inp[i_stride*27] * 1.0171072362820548;
                        } break;
                        case 43: { // l=6, m=7
                            s += inp[i_stride*2] * 2.9131068125936568;
                            s += inp[i_stride*7] * 5.8262136251873136;
                            s += inp[i_stride*9] * -11.6524272503746271;
                            s += inp[i_stride*16] * 2.9131068125936568;
                            s += inp[i_stride*18] * -11.6524272503746271;
                            s += inp[i_stride*20] * 4.6609709001498505;
                        } break;
                        case 44: { // l=6, m=8
                            s += inp[i_stride*0] * 0.4606026297574618;
                            s += inp[i_stride*3] * 0.4606026297574618;
                            s += inp[i_stride*5] * -7.3696420761193888;
                            s += inp[i_stride*10] * -0.4606026297574618;
                            s += inp[i_stride*14] * 7.3696420761193888;
                            s += inp[i_stride*21] * -0.4606026297574618;
                            s += inp[i_stride*23] * 7.3696420761193888;
                            s += inp[i_stride*25] * -7.3696420761193888;
                        } break;
                        case 45: { // l=6, m=9
                            s += inp[i_stride*2] * -2.7636157785447706;
                            s += inp[i_stride*7] * 5.5272315570895412;
                            s += inp[i_stride*9] * 7.3696420761193888;
                            s += inp[i_stride*16] * 8.2908473356343109;
                            s += inp[i_stride*18] * -22.1089262283581647;
                        } break;
                        case 46: { // l=6, m=10
                            s += inp[i_stride*0] * -0.5045649007287241;
                            s += inp[i_stride*3] * 2.5228245036436201;
                            s += inp[i_stride*5] * 5.0456490072872420;
                            s += inp[i_stride*10] * 2.5228245036436201;
                            s += inp[i_stride*12] * -30.2738940437234518;
                            s += inp[i_stride*21] * -0.5045649007287241;
                            s += inp[i_stride*23] * 5.0456490072872420;
                        } break;
                        case 47: { // l=6, m=11
                            s += inp[i_stride*2] * 2.3666191622317525;
                            s += inp[i_stride*7] * -23.6661916223175268;
                            s += inp[i_stride*16] * 11.8330958111587634;
                        } break;
                        case 48: { // l=6, m=12
                            s += inp[i_stride*0] * 0.6831841051919144;
                            s += inp[i_stride*3] * -10.2477615778787161;
                            s += inp[i_stride*10] * 10.2477615778787161;
                            s += inp[i_stride*21] * -0.6831841051919144;
                        } break;
                        }
                        // cart2sph for j
                        switch (j+nfj*lj/3) {
                        case 0: { // l=0, j=0
                            sph_out[0*j_stride] += s * 1;
                        } break;
                        case 1: { // l=1, j=0
                            sph_out[0*j_stride] += s * 1;
                        } break;
                        case 2: { // l=1, j=1
                            sph_out[1*j_stride] += s * 1;
                        } break;
                        case 3: { // l=1, j=2
                            sph_out[2*j_stride] += s * 1;
                        } break;
                        case 4: { // l=2, j=0
                            sph_out[2*j_stride] += s * -0.315391565252520002;
                            sph_out[4*j_stride] += s * 0.546274215296039535;
                        } break;
                        case 5: { // l=2, j=1
                            sph_out[0*j_stride] += s * 1.092548430592079070;
                        } break;
                        case 6: { // l=2, j=2
                            sph_out[3*j_stride] += s * 1.092548430592079070;
                        } break;
                        case 7: { // l=2, j=3
                            sph_out[2*j_stride] += s * -0.315391565252520002;
                            sph_out[4*j_stride] += s * -0.546274215296039535;
                        } break;
                        case 8: { // l=2, j=4
                            sph_out[1*j_stride] += s * 1.092548430592079070;
                        } break;
                        case 9: { // l=2, j=5
                            sph_out[2*j_stride] += s * 0.630783130505040012;
                        } break;
                        case 10: { // l=3, j=0
                            sph_out[4*j_stride] += s * -0.457045799464465739;
                            sph_out[6*j_stride] += s * 0.590043589926643510;
                        } break;
                        case 11: { // l=3, j=1
                            sph_out[0*j_stride] += s * 1.770130769779930531;
                            sph_out[2*j_stride] += s * -0.457045799464465739;
                        } break;
                        case 12: { // l=3, j=2
                            sph_out[3*j_stride] += s * -1.119528997770346170;
                            sph_out[5*j_stride] += s * 1.445305721320277020;
                        } break;
                        case 13: { // l=3, j=3
                            sph_out[4*j_stride] += s * -0.457045799464465739;
                            sph_out[6*j_stride] += s * -1.770130769779930530;
                        } break;
                        case 14: { // l=3, j=4
                            sph_out[1*j_stride] += s * 2.890611442640554055;
                        } break;
                        case 15: { // l=3, j=5
                            sph_out[4*j_stride] += s * 1.828183197857862944;
                        } break;
                        case 16: { // l=3, j=6
                            sph_out[0*j_stride] += s * -0.590043589926643510;
                            sph_out[2*j_stride] += s * -0.457045799464465739;
                        } break;
                        case 17: { // l=3, j=7
                            sph_out[3*j_stride] += s * -1.119528997770346170;
                            sph_out[5*j_stride] += s * -1.445305721320277020;
                        } break;
                        case 18: { // l=3, j=8
                            sph_out[2*j_stride] += s * 1.828183197857862944;
                        } break;
                        case 19: { // l=3, j=9
                            sph_out[3*j_stride] += s * 0.746352665180230782;
                        } break;
                        case 20: { // l=4, j=0
                            sph_out[4*j_stride] += s * 0.317356640745612911;
                            sph_out[6*j_stride] += s * -0.473087347878780002;
                            sph_out[8*j_stride] += s * 0.625835735449176134;
                        } break;
                        case 21: { // l=4, j=1
                            sph_out[0*j_stride] += s * 2.503342941796704538;
                            sph_out[2*j_stride] += s * -0.946174695757560014;
                        } break;
                        case 22: { // l=4, j=2
                            sph_out[5*j_stride] += s * -2.007139630671867500;
                            sph_out[7*j_stride] += s * 1.770130769779930531;
                        } break;
                        case 23: { // l=4, j=3
                            sph_out[4*j_stride] += s * 0.634713281491225822;
                            sph_out[8*j_stride] += s * -3.755014412695056800;
                        } break;
                        case 24: { // l=4, j=4
                            sph_out[1*j_stride] += s * 5.310392309339791593;
                            sph_out[3*j_stride] += s * -2.007139630671867500;
                        } break;
                        case 25: { // l=4, j=5
                            sph_out[4*j_stride] += s * -2.538853125964903290;
                            sph_out[6*j_stride] += s * 2.838524087272680054;
                        } break;
                        case 26: { // l=4, j=6
                            sph_out[0*j_stride] += s * -2.503342941796704530;
                            sph_out[2*j_stride] += s * -0.946174695757560014;
                        } break;
                        case 27: { // l=4, j=7
                            sph_out[5*j_stride] += s * -2.007139630671867500;
                            sph_out[7*j_stride] += s * -5.310392309339791590;
                        } break;
                        case 28: { // l=4, j=8
                            sph_out[2*j_stride] += s * 5.677048174545360108;
                        } break;
                        case 29: { // l=4, j=9
                            sph_out[5*j_stride] += s * 2.676186174229156671;
                        } break;
                        case 30: { // l=4, j=10
                            sph_out[4*j_stride] += s * 0.317356640745612911;
                            sph_out[6*j_stride] += s * 0.473087347878780009;
                            sph_out[8*j_stride] += s * 0.625835735449176134;
                        } break;
                        case 31: { // l=4, j=11
                            sph_out[1*j_stride] += s * -1.770130769779930530;
                            sph_out[3*j_stride] += s * -2.007139630671867500;
                        } break;
                        case 32: { // l=4, j=12
                            sph_out[4*j_stride] += s * -2.538853125964903290;
                            sph_out[6*j_stride] += s * -2.838524087272680050;
                        } break;
                        case 33: { // l=4, j=13
                            sph_out[3*j_stride] += s * 2.676186174229156671;
                        } break;
                        case 34: { // l=4, j=14
                            sph_out[4*j_stride] += s * 0.846284375321634430;
                        } break;
                        case 35: { // l=5, j=0
                            sph_out[6*j_stride] += s * 0.452946651195696921;
                            sph_out[8*j_stride] += s * -0.489238299435250389;
                            sph_out[10*j_stride] += s * 0.656382056840170102;
                        } break;
                        case 36: { // l=5, j=1
                            sph_out[0*j_stride] += s * 3.281910284200850514;
                            sph_out[2*j_stride] += s * -1.467714898305751160;
                            sph_out[4*j_stride] += s * 0.452946651195696921;
                        } break;
                        case 37: { // l=5, j=2
                            sph_out[5*j_stride] += s * 1.754254836801353946;
                            sph_out[7*j_stride] += s * -2.396768392486661870;
                            sph_out[9*j_stride] += s * 2.075662314881041278;
                        } break;
                        case 38: { // l=5, j=3
                            sph_out[6*j_stride] += s * 0.905893302391393842;
                            sph_out[8*j_stride] += s * 0.978476598870500775;
                            sph_out[10*j_stride] += s * -6.563820568401701020;
                        } break;
                        case 39: { // l=5, j=4
                            sph_out[1*j_stride] += s * 8.302649259524165115;
                            sph_out[3*j_stride] += s * -4.793536784973323750;
                        } break;
                        case 40: { // l=5, j=5
                            sph_out[6*j_stride] += s * -5.435359814348363050;
                            sph_out[8*j_stride] += s * 3.913906395482003101;
                        } break;
                        case 41: { // l=5, j=6
                            sph_out[0*j_stride] += s * -6.563820568401701020;
                            sph_out[2*j_stride] += s * -0.978476598870500779;
                            sph_out[4*j_stride] += s * 0.905893302391393842;
                        } break;
                        case 42: { // l=5, j=7
                            sph_out[5*j_stride] += s * 3.508509673602707893;
                            sph_out[9*j_stride] += s * -12.453973889286247600;
                        } break;
                        case 43: { // l=5, j=8
                            sph_out[2*j_stride] += s * 11.741719186446009300;
                            sph_out[4*j_stride] += s * -5.435359814348363050;
                        } break;
                        case 44: { // l=5, j=9
                            sph_out[5*j_stride] += s * -4.678012898136943850;
                            sph_out[7*j_stride] += s * 4.793536784973323755;
                        } break;
                        case 45: { // l=5, j=10
                            sph_out[6*j_stride] += s * 0.452946651195696921;
                            sph_out[8*j_stride] += s * 1.467714898305751163;
                            sph_out[10*j_stride] += s * 3.281910284200850514;
                        } break;
                        case 46: { // l=5, j=11
                            sph_out[1*j_stride] += s * -8.302649259524165110;
                            sph_out[3*j_stride] += s * -4.793536784973323750;
                        } break;
                        case 47: { // l=5, j=12
                            sph_out[6*j_stride] += s * -5.435359814348363050;
                            sph_out[8*j_stride] += s * -11.741719186446009300;
                        } break;
                        case 48: { // l=5, j=13
                            sph_out[3*j_stride] += s * 9.587073569946647510;
                        } break;
                        case 49: { // l=5, j=14
                            sph_out[6*j_stride] += s * 3.623573209565575370;
                        } break;
                        case 50: { // l=5, j=15
                            sph_out[0*j_stride] += s * 0.656382056840170102;
                            sph_out[2*j_stride] += s * 0.489238299435250387;
                            sph_out[4*j_stride] += s * 0.452946651195696921;
                        } break;
                        case 51: { // l=5, j=16
                            sph_out[5*j_stride] += s * 1.754254836801353946;
                            sph_out[7*j_stride] += s * 2.396768392486661877;
                            sph_out[9*j_stride] += s * 2.075662314881041278;
                        } break;
                        case 52: { // l=5, j=17
                            sph_out[2*j_stride] += s * -3.913906395482003100;
                            sph_out[4*j_stride] += s * -5.435359814348363050;
                        } break;
                        case 53: { // l=5, j=18
                            sph_out[5*j_stride] += s * -4.678012898136943850;
                            sph_out[7*j_stride] += s * -4.793536784973323750;
                        } break;
                        case 54: { // l=5, j=19
                            sph_out[4*j_stride] += s * 3.623573209565575370;
                        } break;
                        case 55: { // l=5, j=20
                            sph_out[5*j_stride] += s * 0.935602579627388771;
                        } break;
                        case 56: { // l=6, j=0
                            sph_out[6*j_stride] += s * -0.3178460113381421;
                            sph_out[8*j_stride] += s * 0.4606026297574618;
                            sph_out[10*j_stride] += s * -0.5045649007287241;
                            sph_out[12*j_stride] += s * 0.6831841051919144;
                        } break;
                        case 57: { // l=6, j=1
                            sph_out[0*j_stride] += s * 4.0991046311514863;
                            sph_out[2*j_stride] += s * -2.0182596029148963;
                            sph_out[4*j_stride] += s * 0.9212052595149236;
                        } break;
                        case 58: { // l=6, j=2
                            sph_out[7*j_stride] += s * 2.9131068125936568;
                            sph_out[9*j_stride] += s * -2.7636157785447706;
                            sph_out[11*j_stride] += s * 2.3666191622317525;
                        } break;
                        case 59: { // l=6, j=3
                            sph_out[6*j_stride] += s * -0.9535380340144264;
                            sph_out[8*j_stride] += s * 0.4606026297574618;
                            sph_out[10*j_stride] += s * 2.5228245036436201;
                            sph_out[12*j_stride] += s * -10.2477615778787161;
                        } break;
                        case 60: { // l=6, j=4
                            sph_out[1*j_stride] += s * 11.8330958111587634;
                            sph_out[3*j_stride] += s * -8.2908473356343109;
                            sph_out[5*j_stride] += s * 2.9131068125936568;
                        } break;
                        case 61: { // l=6, j=5
                            sph_out[6*j_stride] += s * 5.7212282040865583;
                            sph_out[8*j_stride] += s * -7.3696420761193888;
                            sph_out[10*j_stride] += s * 5.0456490072872420;
                        } break;
                        case 62: { // l=6, j=6
                            sph_out[0*j_stride] += s * -13.6636821038382887;
                            sph_out[4*j_stride] += s * 1.8424105190298472;
                        } break;
                        case 63: { // l=6, j=7
                            sph_out[7*j_stride] += s * 5.8262136251873136;
                            sph_out[9*j_stride] += s * 5.5272315570895412;
                            sph_out[11*j_stride] += s * -23.6661916223175268;
                        } break;
                        case 64: { // l=6, j=8
                            sph_out[2*j_stride] += s * 20.1825960291489679;
                            sph_out[4*j_stride] += s * -14.7392841522387776;
                        } break;
                        case 65: { // l=6, j=9
                            sph_out[7*j_stride] += s * -11.6524272503746271;
                            sph_out[9*j_stride] += s * 7.3696420761193888;
                        } break;
                        case 66: { // l=6, j=10
                            sph_out[6*j_stride] += s * -0.9535380340144264;
                            sph_out[8*j_stride] += s * -0.4606026297574618;
                            sph_out[10*j_stride] += s * 2.5228245036436201;
                            sph_out[12*j_stride] += s * 10.2477615778787161;
                        } break;
                        case 67: { // l=6, j=11
                            sph_out[1*j_stride] += s * -23.6661916223175268;
                            sph_out[3*j_stride] += s * -5.5272315570895412;
                            sph_out[5*j_stride] += s * 5.8262136251873136;
                        } break;
                        case 68: { // l=6, j=12
                            sph_out[6*j_stride] += s * 11.4424564081731166;
                            sph_out[10*j_stride] += s * -30.2738940437234518;
                        } break;
                        case 69: { // l=6, j=13
                            sph_out[3*j_stride] += s * 22.1089262283581647;
                            sph_out[5*j_stride] += s * -11.6524272503746271;
                        } break;
                        case 70: { // l=6, j=14
                            sph_out[6*j_stride] += s * -7.6283042721154111;
                            sph_out[8*j_stride] += s * 7.3696420761193888;
                        } break;
                        case 71: { // l=6, j=15
                            sph_out[0*j_stride] += s * 4.0991046311514863;
                            sph_out[2*j_stride] += s * 2.0182596029148963;
                            sph_out[4*j_stride] += s * 0.9212052595149236;
                        } break;
                        case 72: { // l=6, j=16
                            sph_out[7*j_stride] += s * 2.9131068125936568;
                            sph_out[9*j_stride] += s * 8.2908473356343109;
                            sph_out[11*j_stride] += s * 11.8330958111587634;
                        } break;
                        case 73: { // l=6, j=17
                            sph_out[2*j_stride] += s * -20.1825960291489679;
                            sph_out[4*j_stride] += s * -14.7392841522387776;
                        } break;
                        case 74: { // l=6, j=18
                            sph_out[7*j_stride] += s * -11.6524272503746271;
                            sph_out[9*j_stride] += s * -22.1089262283581647;
                        } break;
                        case 75: { // l=6, j=19
                            sph_out[4*j_stride] += s * 14.7392841522387776;
                        } break;
                        case 76: { // l=6, j=20
                            sph_out[7*j_stride] += s * 4.6609709001498505;
                        } break;
                        case 77: { // l=6, j=21
                            sph_out[6*j_stride] += s * -0.3178460113381421;
                            sph_out[8*j_stride] += s * -0.4606026297574618;
                            sph_out[10*j_stride] += s * -0.5045649007287241;
                            sph_out[12*j_stride] += s * -0.6831841051919144;
                        } break;
                        case 78: { // l=6, j=22
                            sph_out[1*j_stride] += s * 2.3666191622317525;
                            sph_out[3*j_stride] += s * 2.7636157785447706;
                            sph_out[5*j_stride] += s * 2.9131068125936568;
                        } break;
                        case 79: { // l=6, j=23
                            sph_out[6*j_stride] += s * 5.7212282040865583;
                            sph_out[8*j_stride] += s * 7.3696420761193888;
                            sph_out[10*j_stride] += s * 5.0456490072872420;
                        } break;
                        case 80: { // l=6, j=24
                            sph_out[3*j_stride] += s * -7.3696420761193888;
                            sph_out[5*j_stride] += s * -11.6524272503746271;
                        } break;
                        case 81: { // l=6, j=25
                            sph_out[6*j_stride] += s * -7.6283042721154111;
                            sph_out[8*j_stride] += s * -7.3696420761193888;
                        } break;
                        case 82: { // l=6, j=26
                            sph_out[5*j_stride] += s * 4.6609709001498505;
                        } break;
                        case 83: { // l=6, j=27
                            sph_out[6*j_stride] += s * 1.0171072362820548;
                        } break;
                        }
                    }
                }
            }
        }
    }
}

__global__ static
void ovlp_img_counts_kernel(int *img_counts, PBCIntEnvVars envs,
                            float *exps, float *log_coef, float log_cutoff,
                            int permutation_symmetry)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int bas_ij = item.get_global_id(0);
    #else
    int bas_ij = blockIdx.x * blockDim.x + threadIdx.x;
    #endif
    int bvk_nbas = envs.bvk_ncells * envs.nbas;
    int ish = bas_ij / bvk_nbas;
    int jsh = bas_ij - bvk_nbas * ish;
    if (ish >= envs.nbas || jsh >= bvk_nbas) {
        return;
    }
    int ish_cell0 = ish;
    int jsh_cell0 = jsh % envs.nbas;
    if (permutation_symmetry && ish_cell0 < jsh_cell0) {
        return;
    }
    int nimgs = envs.nimgs;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF + ish_cell0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh_cell0*BAS_SLOTS];
    float ai = exps[ish_cell0];
    float aj = exps[jsh_cell0];
    float aij = ai + aj;
    float ai_aij = ai / aij;
    float aj_aij = aj / aij;
    float theta_ij = ai * aj / aij;
    float log_ci = log_coef[ish_cell0];
    float log_cj = log_coef[jsh_cell0];
    float log_cicj = log_ci + log_cj;
    double *ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    double *rj = env + atm[bas[jsh*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    float xjxi = rj[0] - ri[0];
    float yjyi = rj[1] - ri[1];
    float zjzi = rj[2] - ri[2];
    // log(ci*cj * (pi/aij)**1.5)
    float log_fac = log_cicj + 1.717f - 1.5f*logf(aij);
    // An addiitonal factor for Coulomb integrals
    // log_fac += .25 * logf(2./pi * aij)
    log_fac += .25f * logf(0.6366f * aij);
    log_cutoff = log_cutoff - log_fac;

    int counts = 0;
    for (int img = 0; img < nimgs; ++img) {
        float xjLxi = xjxi + img_coords[img*3+0];
        float yjLyi = yjyi + img_coords[img*3+1];
        float zjLzi = zjzi + img_coords[img*3+2];
        float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
        float theta_ij_rr = theta_ij * rr_ij;
        if (theta_ij_rr > REMOTE_THRESHOLD) {
            continue;
        }
        float dr = sqrtf(rr_ij);
        float dri = aj_aij * dr;
        float drj = ai_aij * dr;
        float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
        float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
        float estimator = dri_fac + drj_fac - theta_ij_rr;
        if (estimator > log_cutoff) {
            counts++;
        }
    }
    img_counts[bas_ij] = counts;
}

__global__ static
void ovlp_img_idx_kernel(int *img_idx, uint32_t *img_offsets, uint32_t *bas_ij_idx, int npairs,
                         PBCIntEnvVars envs, float *exps, float *log_coef, float log_cutoff)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int pair_id = item.get_global_id(0);
    #else
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    #endif
    if (pair_id >= npairs) {
        return;
    }
    uint32_t bas_ij = bas_ij_idx[pair_id];
    int bvk_nbas = envs.bvk_ncells * envs.nbas;
    int ish = bas_ij / bvk_nbas;
    int jsh = bas_ij - bvk_nbas * ish;
    int ish_cell0 = ish;
    int jsh_cell0 = jsh % envs.nbas;
    int nimgs = envs.nimgs;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF + ish_cell0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh_cell0*BAS_SLOTS];
    float ai = exps[ish_cell0];
    float aj = exps[jsh_cell0];
    float aij = ai + aj;
    float ai_aij = ai / aij;
    float aj_aij = aj / aij;
    float theta_ij = ai * aj / aij;
    float log_ci = log_coef[ish_cell0];
    float log_cj = log_coef[jsh_cell0];
    float log_cicj = log_ci + log_cj;
    double *ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    double *rj = env + atm[bas[jsh*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    float xjxi = rj[0] - ri[0];
    float yjyi = rj[1] - ri[1];
    float zjzi = rj[2] - ri[2];
    // log(ci*cj * (pi/aij)**1.5)
    float log_fac = log_cicj + 1.717f - 1.5f*logf(aij);
    // An addiitonal factor for Coulomb integrals
    // log_fac += .25 * logf(2./pi * aij)
    log_fac += .25f * logf(0.6366f * aij);
    log_cutoff = log_cutoff - log_fac;

    int counts = img_offsets[pair_id];
    for (int img = 0; img < nimgs; ++img) {
        float xjLxi = xjxi + img_coords[img*3+0];
        float yjLyi = yjyi + img_coords[img*3+1];
        float zjLzi = zjzi + img_coords[img*3+2];
        float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
        float theta_ij_rr = theta_ij * rr_ij;
        if (theta_ij_rr > REMOTE_THRESHOLD) {
            continue;
        }
        float dr = sqrtf(rr_ij);
        float dri = aj_aij * dr;
        float drj = ai_aij * dr;
        float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
        float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
        float estimator = dri_fac + drj_fac - theta_ij_rr;
        if (estimator > log_cutoff) {
            img_idx[counts] = img;
            counts++;
        }
    }
}

extern "C" {
int PBCsr_int3c2e_latsum23(double *out, PBCIntEnvVars *envs, uint32_t *pool,
                           int shm_size, int nshl_pair, int nbatches_ksh,
                           uint32_t *bas_ij_idx, int *ksh_offsets, int *ksh_idx,
                           int *img_idx, uint32_t *sp_img_offsets,
                           int *gout_stride_lookup, int *ao_pair_loc,
                           int ao_pair_offset, int aux_offset, int bvk_naux, int to_sph,
                           float *diffuse_exps, float *diffuse_coefs,
                           float *atom_aux_exps, float log_cutoff)
{
    #ifdef USE_SYCL
    sycl::range<2> threads(1, THREADS);
    sycl::range<2> blocks(nbatches_ksh, nshl_pair);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class pbc_int3c2e_latsum23_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
        pbc_int3c2e_latsum23_kernel(
            out, dev_envs, pool, bas_ij_idx, ksh_offsets, ksh_idx,
            img_idx, sp_img_offsets, gout_stride_lookup, ao_pair_loc,
            ao_pair_offset, aux_offset, bvk_naux, to_sph,
            diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff,
            item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaFuncSetAttribute(pbc_int3c2e_latsum23_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nshl_pair, nbatches_ksh);
    pbc_int3c2e_latsum23_kernel<<<blocks, THREADS, shm_size>>>(
            out, *envs, pool, bas_ij_idx, ksh_offsets, ksh_idx,
            img_idx, sp_img_offsets, gout_stride_lookup, ao_pair_loc,
            ao_pair_offset, aux_offset, bvk_naux, to_sph,
            diffuse_exps, diffuse_coefs, atom_aux_exps, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int bvk_ovlp_img_counts(int *img_counts, PBCIntEnvVars *envs,
                        float *exps, float *log_coef, float log_cutoff,
                        int permutation_symmetry)
{
    constexpr int threads = 512;
    int bvk_nbas = envs->nbas * envs->bvk_ncells;
    int nbatches = (envs->nbas * bvk_nbas + threads-1) / threads;
    #ifdef USE_SYCL
    auto dev_envs = *envs;
    sycl_get_queue()->parallel_for<class ovlp_img_counts_sycl>(sycl::nd_range<1>(nbatches * threads, threads), [=](auto item) {
      ovlp_img_counts_kernel(img_counts, dev_envs, exps, log_coef, log_cutoff, permutation_symmetry);
    });
    #else
    ovlp_img_counts_kernel<<<nbatches, threads>>>(
            img_counts, *envs, exps, log_coef, log_cutoff, permutation_symmetry);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bvk_ovlp_img_counts: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int bvk_ovlp_img_idx(int *img_idx, uint32_t *img_offsets, uint32_t *bas_ij_idx, int npairs,
                     PBCIntEnvVars *envs, float *exps, float *log_coef, float log_cutoff)
{
    constexpr int threads = 512;
    int blocks = (npairs + threads-1) / threads;
    #ifdef USE_SYCL
    auto dev_envs = *envs;
    sycl_get_queue()->parallel_for<class ovlp_img_idx_sycl>(sycl::nd_range<1>(blocks * threads, threads), [=](auto item) {
      ovlp_img_idx_kernel(img_idx, img_offsets, bas_ij_idx, npairs, dev_envs, exps, log_coef, log_cutoff);
    });
    #else
    ovlp_img_idx_kernel<<<blocks, threads>>>(
        img_idx, img_offsets, bas_ij_idx, npairs, *envs, exps, log_coef, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bvk_ovlp_img_idx: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
