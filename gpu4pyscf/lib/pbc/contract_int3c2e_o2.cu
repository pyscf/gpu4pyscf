/*
 * Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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
#include <cub/cub.cuh>
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "int3c2e_create_tasks_o2.cuh"

#define NF_AUX_MAX      28
#define GOUT_WIDTH      30

// lattice sum over j and k for (ij|k)
__global__ static
void contract_int3c2e_dm_kernel(double *out, double *dm, PBCIntEnvVars envs,
                                uint32_t *img_pool, ShellTripletTaskInfo *task_pool,
                                uint32_t *bas_ij_idx, int *shl_pair_offsets,
                                int *img_idx, uint32_t *sp_img_offsets,
                                int *gout_stride_lookup, int nauxbas,
                                float *diffuse_exps, float *diffuse_coefs, float log_cutoff,
                                int *head, int sp_blocks)
{
    int thread_id = threadIdx.x;
    __shared__ int sp_block_id, ksh_cell0;
    img_pool += blockIdx.x * POOL_SIZE * (MAX_IMGS_PER_TASK+2);
    // rem_task_idx stores the Id of the ijk tasks which has remaining_imgs > 0
    uint32_t *rem_task_idx = img_pool + POOL_SIZE * MAX_IMGS_PER_TASK;
    uint32_t *sub_task_idx = img_pool + POOL_SIZE *(MAX_IMGS_PER_TASK+1);
    ShellTripletTaskInfo *ijk_tasks_info = task_pool + blockIdx.x * POOL_SIZE;
while (1) {
    if (thread_id == 0) {
        int batch_id = atomicAdd(head, 1);
        sp_block_id = batch_id / nauxbas;
        ksh_cell0 = batch_id % nauxbas;
    }
    __syncthreads();
    if (sp_block_id >= sp_blocks) {
        return;
    }

    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int nao;
    __shared__ int gout_stride, nst_per_block;
    __shared__ int expk, ck;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij = bas_ij_idx[shl_pair0];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int ksh = bvk_nbas + ksh_cell0;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int lij = li + lj;
        nroots = ((lij + lk) / 2 + 1) * 2;
        iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[envs.nbas];
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfk = c_nf[lk];
    int nfij = nfi * nfj;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nst_per_block;
    extern __shared__ double shared_memory[];
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

    double vj_xyz[NF_AUX_MAX];
    for (int n = 0; n < NF_AUX_MAX; ++n) {
        vj_xyz[n] = 0;
    }

while (shl_pair0 < shl_pair1) {
    int n_pairs = min(POOL_SIZE/ncells, shl_pair1 - shl_pair0);
    initialize_ijk_tasks(img_pool, rem_task_idx, ijk_tasks_info, envs,
                         shl_pair0, shl_pair0+n_pairs, ksh_cell0, ksh_cell0+1,
                         li, lj, nauxbas, bas_ij_idx, img_idx, sp_img_offsets,
                         diffuse_exps, diffuse_coefs, log_cutoff);
    __shared__ int num_ijk_tasks;
    if (thread_id == 0) {
        num_ijk_tasks = ncells * n_pairs;
    }
    __syncthreads();
    while (num_ijk_tasks > 0) {
    _filter_jk_images(img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info, envs, img_idx);
    __shared__ int num_sub_tasks, img_count_lower;
    if (thread_id == 0) {
        img_count_lower = 1;
    }
    __syncthreads();
    while (img_count_lower <= MAX_IMGS_PER_TASK) {
        _select_sub_tasks(sub_task_idx, num_sub_tasks, img_count_lower,
                          rem_task_idx, num_ijk_tasks, nst_per_block, ijk_tasks_info);
        if (num_sub_tasks == 0) continue;
        for (int task_id = st_id; task_id < num_sub_tasks + st_id; task_id += nst_per_block) {
            ShellTripletTaskInfo *ijk_task = ijk_tasks_info;
            int ijk_id = 0;
            int img_count = 0;
            if (task_id < num_sub_tasks) {
                ijk_id = sub_task_idx[task_id];
                ijk_task += ijk_id;
                img_count = ijk_task->img_count;
            }
            __shared__ int max_img_count;
            block_max(img_count, max_img_count);

            int ksh = ijk_task->ksh;
            int pair_ij = ijk_task->pair_ij;
            uint32_t bas_ij = bas_ij_idx[pair_ij];
            int ish = bas_ij / bvk_nbas;
            int jsh = bas_ij - bvk_nbas * ish;
            int ish_cell0 = ish;
            int jsh_cell0 = jsh % envs.nbas;
            int i0 = envs.ao_loc[ish];
            int j0 = envs.ao_loc[jsh];
            double *dm_local = dm + j0 * nao + i0;
            // TODO: load dm_local to register
            int expi = bas[ish_cell0*BAS_SLOTS+PTR_EXP];
            int expj = bas[jsh_cell0*BAS_SLOTS+PTR_EXP];
            int ci = bas[ish_cell0*BAS_SLOTS+PTR_COEFF];
            int cj = bas[jsh_cell0*BAS_SLOTS+PTR_COEFF];
            int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

            for (int img = 0; img < max_img_count; img++) {
                int img_jk = 0;
                if (img < img_count) {
                    img_jk = img_pool[ijk_id+POOL_SIZE*img];
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp - jprim * ip;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (gout_id == 0) {
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
                        if (ish_cell0 >= jsh_cell0 && img < img_count && task_id < num_sub_tasks) {
                            double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                            double theta_ij = ai * aj_aij;
                            double Kab = theta_ij * rr_ij;
                            double cicj = PI_FAC * env[ci+ip] * env[cj+jp];
                            if (ish_cell0 == jsh_cell0) {
                                cicj *= .5;
                            }
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
                                if (img < img_count && task_id < num_sub_tasks) {
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
                            if (img < img_count && task_id < num_sub_tasks) {
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
    } // while (img_count_lower <= MAX_IMGS_PER_TASK)
    _filter_ijk_tasks(rem_task_idx, num_ijk_tasks, ijk_tasks_info);
    } // while (num_ijk_tasks > 0)
    if (thread_id == 0) {
        shl_pair0 += POOL_SIZE / ncells;
    }
    __syncthreads();
}

    double *vj = out + envs.ao_loc[bvk_nbas + ksh_cell0] - envs.ao_loc[bvk_nbas];
    typedef cub::BlockReduce<double, THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#pragma unroll
    for (int n = 0; n < NF_AUX_MAX; ++n) {
        if (n >= nfk) break;
        // Reduce within block
        double val = vj_xyz[n];
        double block_sum = BlockReduce(temp_storage).Sum(val);
        if (thread_id == 0) {
            atomicAdd(vj + n, block_sum);
        }
    }
}
}

__global__ static
void contract_int3c2e_auxvec_kernel(double *out, double *auxvec, PBCIntEnvVars envs,
                                    uint32_t *img_pool, ShellTripletTaskInfo *task_pool,
                                    uint32_t *bas_ij_idx, int *ksh_offsets,
                                    int *img_idx, uint32_t *sp_img_offsets,
                                    int *gout_stride_lookup, int nauxbas,
                                    float *diffuse_exps, float *diffuse_coefs, float log_cutoff,
                                    int *head, int npairs_ij, int ksh_blocks)
{
    int thread_id = threadIdx.x;
    __shared__ int pair_ij, ksh_block_id;
    img_pool += blockIdx.x * POOL_SIZE * (MAX_IMGS_PER_TASK+2);
    // rem_task_idx stores the Id of the ijk tasks which has remaining_imgs > 0
    uint32_t *rem_task_idx = img_pool + POOL_SIZE * MAX_IMGS_PER_TASK;
    uint32_t *sub_task_idx = img_pool + POOL_SIZE *(MAX_IMGS_PER_TASK+1);
    ShellTripletTaskInfo *ijk_tasks_info = task_pool + blockIdx.x * POOL_SIZE;
while (1) {
    if (thread_id == 0) {
        int batch_id = atomicAdd(head, 1);
        pair_ij = batch_id / ksh_blocks;
        ksh_block_id = batch_id % ksh_blocks;
    }
    __syncthreads();
    if (pair_ij >= npairs_ij) {
        return;
    }

    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    double omega = env[PTR_RANGE_OMEGA];
    int nimgs = envs.nimgs;
    __shared__ int ksh0_cell0, ksh1_cell0;
    __shared__ int ish, jsh, li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int gout_stride, nst_per_block;
    __shared__ int expi, expj, ci, cj;
    __shared__ double xi, yi, zi, xjxi, yjyi, zjzi;
    __shared__ double fac;
    if (thread_id == 0) {
        ksh0_cell0 = ksh_offsets[ksh_block_id];
        ksh1_cell0 = ksh_offsets[ksh_block_id+1];
        ish = bas_ij / bvk_nbas;
        jsh = bas_ij % bvk_nbas;
        int ksh = bvk_nbas + ksh0_cell0;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int lij = li + lj;
        nroots = ((lij + lk) / 2 + 1) * 2;
        iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
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
    }
    __syncthreads();
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfk = c_nf[lk];
    int nfij = nfi * nfj;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nst_per_block;
    extern __shared__ double shared_memory[];
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

while (ksh0_cell0 < ksh1_cell0) {
    int nksh = min(POOL_SIZE/ncells, ksh1_cell0 - ksh0_cell0);
    initialize_ijk_tasks(img_pool, rem_task_idx, ijk_tasks_info, envs,
                         pair_ij, pair_ij+1, ksh0_cell0, ksh0_cell0+nksh,
                         li, lj, nauxbas, bas_ij_idx, img_idx, sp_img_offsets,
                         diffuse_exps, diffuse_coefs, log_cutoff);
    __shared__ int num_ijk_tasks;
    if (thread_id == 0) {
        num_ijk_tasks = nksh * ncells;
    }
    __syncthreads();
    while (num_ijk_tasks > 0) {
    _filter_jk_images(img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info, envs, img_idx);
    __shared__ int num_sub_tasks, img_count_lower;
    if (thread_id == 0) {
        img_count_lower = 1;
    }
    __syncthreads();
    while (img_count_lower <= MAX_IMGS_PER_TASK) {
        _select_sub_tasks(sub_task_idx, num_sub_tasks, img_count_lower,
                          rem_task_idx, num_ijk_tasks, nst_per_block, ijk_tasks_info);
        if (num_sub_tasks == 0) continue;
        for (int task_id = st_id; task_id < num_sub_tasks + st_id; task_id += nst_per_block) {
            ShellTripletTaskInfo *ijk_task = ijk_tasks_info;
            int ijk_id = 0;
            int img_count = 0;
            if (task_id < num_sub_tasks) {
                ijk_id = sub_task_idx[task_id];
                ijk_task += ijk_id;
                img_count = ijk_task->img_count;
            }
            __shared__ int max_img_count;
            block_max(img_count, max_img_count);

            int ksh = ijk_task->ksh;
            int k_cell_id = (ksh - bvk_nbas) / nauxbas;
            int ksh_cell0 = ksh - k_cell_id * nauxbas;
            int k0 = ao_loc[ksh_cell0] - ao_loc[bvk_nbas];
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

            for (int img = 0; img < max_img_count; img++) {
                int img_jk = 0;
                if (img < img_count) {
                    img_jk = img_pool[ijk_id+POOL_SIZE*img];
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp - jprim * ip;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (gout_id == 0) {
                        int jL = img_jk / nimgs;
                        int kL = img_jk - nimgs * jL;
                        double xjLxi = xjxi + img_coords[jL*3+0];
                        double yjLyi = yjyi + img_coords[jL*3+1];
                        double zjLzi = zjzi + img_coords[jL*3+2];
                        double fac_ij = 0;
                        if (img < img_count && task_id < num_sub_tasks) {
                            double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                            double theta_ij = ai * aj_aij;
                            double Kab = theta_ij * rr_ij;
                            double cicj = fac * env[ci+ip] * env[cj+jp];
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
                        gx[gx_len] = fac_ij;
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
                                if (img < img_count && task_id < num_sub_tasks) {
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
                            if (img < img_count && task_id < num_sub_tasks) {
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
    } // while (img_count_lower <= MAX_IMGS_PER_TASK)
    _filter_ijk_tasks(rem_task_idx, num_ijk_tasks, ijk_tasks_info);
    } // while (num_ijk_tasks > 0)
    if (thread_id == 0) {
        ksh0_cell0 += nksh;
    }
    __syncthreads();
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
}

extern "C" {
int PBCcontract_int3c2e_dm_o2(double *out, double *dm, PBCIntEnvVars *envs,
                           uint32_t *pool, ShellTripletTaskInfo *task_pool, int *head,
                           int shm_size, int nbatches_shl_pair, int nauxbas,
                           uint32_t *bas_ij_idx, int *shl_pair_offsets,
                           int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                           float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    cudaFuncSetAttribute(contract_int3c2e_dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    cudaMemset(head, 0, sizeof(int));
    contract_int3c2e_dm_kernel<<<workers, THREADS, shm_size>>>(
            out, dm, *envs, pool, task_pool, bas_ij_idx, shl_pair_offsets,
            img_idx, img_offsets, gout_stride_lookup, nauxbas,
            diffuse_exps, diffuse_coefs, log_cutoff,
            head, nbatches_shl_pair);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCcontract_int3c2e_auxvec_o2(double *out, double *auxvec, PBCIntEnvVars *envs,
                               uint32_t *pool, ShellTripletTaskInfo *task_pool, int *head,
                               int shm_size, int npairs, int nbatches_ksh, int nauxbas,
                               uint32_t *bas_ij_idx, int *ksh_offsets,
                               int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                               float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    cudaFuncSetAttribute(contract_int3c2e_auxvec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    cudaMemset(head, 0, sizeof(int));
    contract_int3c2e_auxvec_kernel<<<workers, THREADS, shm_size>>>(
            out, auxvec, *envs, pool, task_pool, bas_ij_idx, ksh_offsets,
            img_idx, img_offsets, gout_stride_lookup, nauxbas,
            diffuse_exps, diffuse_coefs, log_cutoff,
            head, npairs, nbatches_ksh);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
