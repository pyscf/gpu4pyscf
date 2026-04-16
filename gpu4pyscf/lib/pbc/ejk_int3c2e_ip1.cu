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
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "int3c2e_create_tasks.cuh"

#define GOUT_WIDTH      54

__global__ static
void ejk_int3c2e_ip1_kernel(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                            PBCIntEnvVars envs, uint32_t *pool, ShellTripletTaskInfo *task_pool,
                            uint32_t *bas_ij_idx, int *shl_pair_offsets, int *ksh_offsets,
                            int *img_idx, uint32_t *sp_img_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int aux_offset, int nauxbas, int naux,
                            float *diffuse_exps, float *diffuse_coefs, float log_cutoff,
                            int *head, int sp_blocks, int ksh_blocks)
{
    int thread_id = threadIdx.x;
    __shared__ int sp_block_id, ksh_block_id;
    uint32_t *img_pool = pool + blockIdx.x * POOL_SIZE * (MAX_IMGS_PER_TASK+2);
    uint32_t *rem_task_idx = img_pool + POOL_SIZE * MAX_IMGS_PER_TASK;
    uint32_t *sub_task_idx = img_pool + POOL_SIZE *(MAX_IMGS_PER_TASK+1);
    ShellTripletTaskInfo *ijk_tasks_info = task_pool + blockIdx.x * POOL_SIZE;
while (1) {
    if (thread_id == 0) {
        int batch_id = atomicAdd(head, 1);
        sp_block_id = batch_id / ksh_blocks;
        ksh_block_id = batch_id % ksh_blocks;
    }
    __syncthreads();
    if (sp_block_id >= sp_blocks) {
        return;
    }

    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    __shared__ int ksh0_cell0, ksh1_cell0;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nao;
    __shared__ int g_size, gout_stride, nst_per_block;
    if (thread_id == 0) {
        int bvk_nbas = envs.nbas * ncells;
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        ksh0_cell0 = ksh_offsets[ksh_block_id];
        ksh1_cell0 = ksh_offsets[ksh_block_id+1];
        uint32_t bas_ij = bas_ij_idx[shl_pair0];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int ksh = bvk_nbas + ksh0_cell0;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int lij = li + lj + 1;
        nroots = ((lij + lk) / 2 + 1) * 2;
        iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[envs.cell0_nbas];
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
    }
    __syncthreads();

    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    __shared__ int num_ijk_tasks;
    if (thread_id == 0) {
        int nshl_pairs = shl_pair1 - shl_pair0;
        int nksh = ksh1_cell0 - ksh0_cell0;
        num_ijk_tasks = nksh * ncells * nshl_pairs;
    }
    __syncthreads();
    initialize_ijk_tasks(img_pool, rem_task_idx, ijk_tasks_info, envs,
                         shl_pair0, shl_pair1, ksh0_cell0, ksh1_cell0,
                         li, lj, lk, nauxbas, bas_ij_idx, img_idx, sp_img_offsets,
                         diffuse_exps, diffuse_coefs, log_cutoff);
    while (num_ijk_tasks > 0) {
    _filter_jk_images(img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info,
                      envs, img_idx);
    __shared__ int num_sub_tasks, img_not_processed, img_tile_size;
    if (thread_id == 0) {
        img_tile_size = 8;
        img_not_processed = MAX_IMGS_PER_TASK;
    }
    __syncthreads();
    while (img_not_processed > 0) {
        _select_sub_ijk(sub_task_idx, num_sub_tasks, img_not_processed, img_tile_size,
                        rem_task_idx, num_ijk_tasks, ijk_tasks_info);
        for (int task_id = st_id; task_id < num_sub_tasks + st_id; task_id += nst_per_block) {
            ShellTripletTaskInfo *ijk_task = ijk_tasks_info;
            int ijk_id = 0;
            int img_start = 0;
            if (task_id < num_sub_tasks) {
                ijk_id = sub_task_idx[task_id];
                ijk_task += ijk_id;
                img_start = ijk_task->img_count;
            }
            int ksh = ijk_task->ksh;
            int pair_ij = ijk_task->pair_ij;
            uint32_t bas_ij = bas_ij_idx[pair_ij];
            int bvk_nbas = envs.nbas * ncells;
            int ish = bas_ij / bvk_nbas;
            int jsh = bas_ij - bvk_nbas * ish;
            int ish_cell0 = ish;
            int jsh_cell0 = jsh % envs.nbas;
            double fac = PI_FAC;
            if (ish_cell0 == jsh_cell0) {
                fac *= .5;
            } else if (ish_cell0 < jsh_cell0) {
                fac = 0;
            }
            int k_cell_id = (ksh - bvk_nbas) / nauxbas;
            int ksh_cell0 = ksh - k_cell_id * nauxbas;
            int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
            int nfi = c_nf[li];
            int nfj = c_nf[lj];
            float div_nfi = c_div_nf[li];
            float div_nfj = c_div_nf[lj];
            double dm_tensor[GOUT_WIDTH];
            if (task_id < num_sub_tasks) {
                if (density_auxvec == NULL) {
                    float div_nfij = div_nfi * div_nfj;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    int bvk_naux = naux * ncells;
                    double *dm_local = dm + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        uint32_t ijk = n*gout_stride+gout_id;
                        if (ijk >= nf) break;
                        uint32_t k = ijk * div_nfij;
                        uint32_t ij = ijk - k * nfi*nfj;
                        dm_tensor[n] = dm_local[ij*bvk_naux + k] * fac;
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        uint32_t ijk = n*gout_stride+gout_id;
                        if (ijk >= nf) break;
                        uint32_t jk = ijk * div_nfi;
                        uint32_t i = ijk - jk * nfi;
                        uint32_t k = jk * div_nfj;
                        uint32_t j = jk - k * nfj;
                        dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k] * fac;
                    }
                }
            }

            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            double v_ix = 0;
            double v_iy = 0;
            double v_iz = 0;
            double v_jx = 0;
            double v_jy = 0;
            double v_jz = 0;
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int expi = bas[ish*BAS_SLOTS+PTR_EXP];
                int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                for (int img = 0; img < img_tile_size; img++) {
                    int img_jk = 0;
                    if (task_id < num_sub_tasks) {
                        img_jk = img_pool[ijk_id+POOL_SIZE*(img_start+img)];
                    }
                    __syncthreads();
                    if (gout_id == 0) {
                        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
                        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
                        int jL = img_jk / nimgs;
                        int kL = img_jk - nimgs * jL;
                        double xi = env[ri+0];
                        double yi = env[ri+1];
                        double zi = env[ri+2];
                        double xjxi = env[rj+0] - xi;
                        double yjyi = env[rj+1] - yi;
                        double zjzi = env[rj+2] - zi;
                        double xjLxi = xjxi + img_coords[jL*3+0];
                        double yjLyi = yjyi + img_coords[jL*3+1];
                        double zjLzi = zjzi + img_coords[jL*3+2];
                        double fac_ij = 0;
                        if (task_id < num_sub_tasks) {
                            double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                            double theta_ij = ai * aj_aij;
                            double Kab = theta_ij * rr_ij;
                            double cicj = env[ci+ip] * env[cj+jp];
                            fac_ij = exp(-Kab) * cicj;
                        }
                        double xij = xjLxi * aj_aij + xi;
                        double yij = yjLyi * aj_aij + yi;
                        double zij = zjLzi * aj_aij + zi;
                        double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                        double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                        double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                        rjri[0*nst_per_block] = xjLxi;
                        rjri[1*nst_per_block] = yjLyi;
                        rjri[2*nst_per_block] = zjLzi;
                        Rpq[0*nst_per_block] = xpq;
                        Rpq[1*nst_per_block] = ypq;
                        Rpq[2*nst_per_block] = zpq;
                        Rpq[3*nst_per_block] = rr;
                        int gx_len = g_size * nst_per_block;
                        gx[gx_len] = fac_ij;
                    }
                    for (int kp = 0; kp < kprim; ++kp) {
                        double ak = env[expk+kp];
                        double theta = aij * ak / (aij + ak);
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                        }
                        double omega = env[PTR_RANGE_OMEGA];
                        rys_roots_rs(nroots, theta, Rpq[3*nst_per_block], omega,
                                     rw, nst_per_block, gout_id, gout_stride);
                        for (int irys = 0; irys < nroots; ++irys) {
                            int nst = nst_per_block;
                            int stride_j = li + 2;
                            int stride_k = stride_j * (lj + 1);
                            int gsize = g_size;
                            int gx_len = gsize * nst_per_block;
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
                                if (task_id < num_sub_tasks) {
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
                            if (task_id < num_sub_tasks) {
                                int nfi = c_nf[li];
                                int nfj = c_nf[lj];
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
                                    double fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                                    double fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                                    double fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                                    double fjx = aj2 * (gix - rjri[0*nst] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                                    double fjy = aj2 * (giy - rjri[1*nst] * Iy); if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                                    double fjz = aj2 * (giz - rjri[2*nst] * Iz); if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
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
            }
            __syncthreads();
            int ia = bas[ish*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            double *reduce = shared_memory + thread_id;
            __syncthreads();
//            double v_kx = -v_ix - v_jx;
//            double v_ky = -v_iy - v_jy;
//            double v_kz = -v_iz - v_jz;
            reduce[0*THREADS] = v_kx;
            reduce[1*THREADS] = v_ky;
            reduce[2*THREADS] = v_kz;
            reduce[3*THREADS] = v_ix;
            reduce[4*THREADS] = v_iy;
            reduce[5*THREADS] = v_iz;
            reduce[6*THREADS] = v_jx;
            reduce[7*THREADS] = v_jy;
            reduce[8*THREADS] = v_jz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*THREADS] += reduce[n*THREADS+i*nst_per_block];
                    }
                }
            }
            if (gout_id == 0) {
                atomicAdd(ejk_aux+ka*3+0, reduce[0*THREADS]);
                atomicAdd(ejk_aux+ka*3+1, reduce[1*THREADS]);
                atomicAdd(ejk_aux+ka*3+2, reduce[2*THREADS]);
                atomicAdd(ejk+ia*3+0, reduce[3*THREADS]);
                atomicAdd(ejk+ia*3+1, reduce[4*THREADS]);
                atomicAdd(ejk+ia*3+2, reduce[5*THREADS]);
                atomicAdd(ejk+ja*3+0, reduce[6*THREADS]);
                atomicAdd(ejk+ja*3+1, reduce[7*THREADS]);
                atomicAdd(ejk+ja*3+2, reduce[8*THREADS]);
            }
            __syncthreads();
        }
    } // while (img_not_processed > 0)
    _filter_ijk_tasks(rem_task_idx, num_ijk_tasks, ijk_tasks_info);
    } // while (num_ijk_tasks > 0)
}
}

extern "C" {
int PBCsr_ejk_int3c2e_ip1(double *ejk, double*ejk_aux, double *dm, double *density_auxvec,
                          PBCIntEnvVars *envs, uint32_t *pool,
                          ShellTripletTaskInfo *task_pool, int *head,
                          int shm_size, int nbatches_shl_pair, int nbatches_ksh,
                          uint32_t *bas_ij_idx, int *shl_pair_offsets, int *ksh_offsets,
                          int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                          int *ao_pair_loc, int aux_offset, int nauxbas, int naux,
                          float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    cudaFuncSetAttribute(ejk_int3c2e_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    cudaMemset(head, 0, sizeof(int));
    ejk_int3c2e_ip1_kernel<<<workers, THREADS, shm_size>>>(
            ejk, ejk_aux, dm, density_auxvec, *envs, pool, task_pool,
            bas_ij_idx, shl_pair_offsets, ksh_offsets, img_idx, img_offsets,
            gout_stride_lookup, ao_pair_loc, aux_offset, nauxbas, naux,
            diffuse_exps, diffuse_coefs, log_cutoff,
            head, nbatches_shl_pair, nbatches_ksh);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
