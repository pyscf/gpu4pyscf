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
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "gvhf-rys/build_rys_gxyz.cuh"
#include "int3c2e_create_tasks.cuh"

#define GOUT_WIDTH      29

__global__ static
void contract_int3c2e_pvp_auxvec_kernel(double *out, double *auxvec, PBCIntEnvVars envs,
                                        uint32_t *img_pool, ShellTripletTaskInfo *task_pool,
                                        uint32_t *bas_ij_idx, int *ksh_offsets,
                                        int *img_idx, uint32_t *sp_img_offsets,
                                        int *gout_stride_lookup, int nauxbas,
                                        float *diffuse_exps, float *diffuse_coefs, float log_cutoff,
                                        int *head, int npairs_ij, int ksh_blocks)
{
    int thread_id = threadIdx.x;
    img_pool += blockIdx.x * POOL_SIZE * (MAX_IMGS_PER_TASK+2);
    // rem_task_idx stores the Id of the ijk tasks which has remaining_imgs > 0
    uint32_t *rem_task_idx = img_pool + POOL_SIZE * MAX_IMGS_PER_TASK;
    uint32_t *sub_task_idx = img_pool + POOL_SIZE *(MAX_IMGS_PER_TASK+1);
    ShellTripletTaskInfo *ijk_tasks_info = task_pool + blockIdx.x * POOL_SIZE;
    extern __shared__ double shared_memory[];
    __shared__ int pair_ij, ksh_block_id;
    __shared__ int ksh0_cell0, ksh1_cell0;
    __shared__ int ish, jsh, li, lj, lk, nroots;
    __shared__ int iprim, jprim, kprim;
    __shared__ int gout_stride, nst_per_block;
    __shared__ int expi, expj, ci, cj;
    __shared__ double xi, yi, zi, xjxi, yjyi, zjzi;
    __shared__ int num_ijk_tasks;
    __shared__ int num_sub_tasks, img_not_processed, img_tile_size;
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
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    double omega = env[PTR_RANGE_OMEGA];
    int nimgs = envs.nimgs;
    if (thread_id == 0) {
        int bvk_nbas = envs.nbas * ncells;
        ksh0_cell0 = ksh_offsets[ksh_block_id];
        ksh1_cell0 = ksh_offsets[ksh_block_id+1];
        ish = bas_ij / bvk_nbas;
        jsh = bas_ij % bvk_nbas;
        int ksh = bvk_nbas + ksh0_cell0;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        int lij = li + lj + 2;
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
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int ish_cell0 = ish;
    int jsh_cell0 = jsh % envs.nbas;
    if (ish_cell0 < jsh_cell0) {
        continue;
    }
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id - gout_id * nst_per_block;

    int stride_j = li + 2;
    int stride_k = stride_j * (lj + 2);
    int g_size = stride_k * (lk + 1);
    int gx_len = g_size * nst_per_block;
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 6 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+6) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    double vj[GOUT_WIDTH];
    for (int n = 0; n < GOUT_WIDTH; ++n) {
        vj[n] = 0;
    }

while (ksh0_cell0 < ksh1_cell0) {
    int nksh = min(POOL_SIZE/ncells, ksh1_cell0 - ksh0_cell0);
    initialize_ijk_tasks(img_pool, rem_task_idx, ijk_tasks_info, envs,
                         pair_ij, pair_ij+1, ksh0_cell0, ksh0_cell0+nksh,
                         li, lj, lk, nauxbas, bas_ij_idx, img_idx, sp_img_offsets,
                         diffuse_exps, diffuse_coefs, log_cutoff);
    if (thread_id == 0) {
        num_ijk_tasks = nksh * ncells;
    }
    __syncthreads();
    while (num_ijk_tasks > 0) {
    _filter_jk_images(img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info, envs, img_idx);
    if (thread_id == 0) {
        img_tile_size = 8;
        img_not_processed = MAX_IMGS_PER_TASK;
    }
    __syncthreads();
    while (img_not_processed > 0) {
        _select_sub_ijk(sub_task_idx, num_sub_tasks, img_not_processed, img_tile_size,
                        rem_task_idx, num_ijk_tasks, ijk_tasks_info, (int *)shared_memory);
        if (num_sub_tasks == 0) continue;
        for (int task_id = st_id; task_id < num_sub_tasks + st_id; task_id += nst_per_block) {
            ShellTripletTaskInfo *ijk_task = ijk_tasks_info;
            int ijk_id = 0;
            int img_start = 0;
            if (task_id < num_sub_tasks) {
                ijk_id = sub_task_idx[task_id];
                ijk_task += ijk_id;
                img_start = ijk_task->img_count;
            }
            int bvk_nbas = envs.nbas * ncells;
            int ksh = ijk_task->ksh;
            int k_cell_id = (ksh - bvk_nbas) / nauxbas;
            int ksh_cell0 = ksh - k_cell_id * nauxbas;
            int k0 = ao_loc[ksh_cell0] - ao_loc[bvk_nbas];

            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
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
                        int jL = img_jk / nimgs;
                        int kL = img_jk - nimgs * jL;
                        double xjLxi = xjxi + img_coords[jL*3+0];
                        double yjLyi = yjyi + img_coords[jL*3+1];
                        double zjLzi = zjzi + img_coords[jL*3+2];
                        double fac_ij = 0;
                        if (task_id < num_sub_tasks) {
                            double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                            double theta_ij = ai * aj_aij;
                            double Kab = theta_ij * rr_ij;
                            double cicj = PI_FAC * env[ci+ip] * env[cj+jp];
                            fac_ij = exp(-Kab) * cicj;
                        }
                        double xij = xjLxi * aj_aij + xi;
                        double yij = yjLyi * aj_aij + yi;
                        double zij = zjLzi * aj_aij + zi;
                        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                        double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                        double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                        double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                        rjri[0*nst_per_block] = xjLxi;
                        rjri[1*nst_per_block] = yjLyi;
                        rjri[2*nst_per_block] = zjLzi;
                        Rpq[0*nst_per_block] = xpq;
                        Rpq[1*nst_per_block] = ypq;
                        Rpq[2*nst_per_block] = zpq;
                        gx[gx_len] = fac_ij;
                    }
                    for (int kp = 0; kp < kprim; ++kp) {
                        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
                        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                        double ak = env[expk+kp];
                        double theta = aij * ak / (aij + ak);
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                        }
                        double xpq = Rpq[0*nst_per_block];
                        double ypq = Rpq[1*nst_per_block];
                        double zpq = Rpq[2*nst_per_block];
                        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                        rys_roots_rs(nroots, theta, rr, omega,
                                     rw, nst_per_block, gout_id, gout_stride);
                        for (int irys = 0; irys < nroots; ++irys) {
                            int lij = li + lj + 2;
                            BUILD_3C_GXYZ(lj+1, nst_per_block, task_id < num_sub_tasks);
                            if (task_id < num_sub_tasks) {
                                int nfk = c_nf[lk];
                                for (int k = 0; k < nfk; ++k) {
                                    int kx = _c_cartesian_lexical_xyz[idx_k+k*3+0];
                                    int ky = _c_cartesian_lexical_xyz[idx_k+k*3+1];
                                    int kz = _c_cartesian_lexical_xyz[idx_k+k*3+2];
                                    double rho = auxvec[k0+k];
#pragma unroll
                                    for (int n = 0; n < GOUT_WIDTH; n++) {
                                        int nfi = c_nf[li];
                                        int nfj = c_nf[lj];
                                        int nfij = nfi * nfj;
                                        uint32_t ij = gout_id + n * gout_stride;
                                        if (ij >= nfij) break;
                                        float div_nfi = c_div_nf[li];
                                        uint32_t j = ij * div_nfi;
                                        uint32_t i = ij - nfi * j;
                                        int ix = _c_cartesian_lexical_xyz[idx_i+i*3+0];
                                        int iy = _c_cartesian_lexical_xyz[idx_i+i*3+1];
                                        int iz = _c_cartesian_lexical_xyz[idx_i+i*3+2];
                                        int jx = _c_cartesian_lexical_xyz[idx_j+j*3+0];
                                        int jy = _c_cartesian_lexical_xyz[idx_j+j*3+1];
                                        int jz = _c_cartesian_lexical_xyz[idx_j+j*3+2];
                                        int i_1 =          nst_per_block;
                                        int j_1 = stride_j*nst_per_block;
                                        int addrx = (ix + jx*stride_j + kx*stride_k) * nst_per_block;
                                        int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nst_per_block;
                                        int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nst_per_block;
                                        double ai2 = -2. * ai;
                                        double aj2 = -2. * aj;
                                        double f3x = ai2 * gx[addrx+i_1+j_1];
                                        if (ix > 0) { f3x += ix * gx[addrx-i_1+j_1]; } f3x *= aj2;
                                        if (jx > 0) {
                                            double fx = ai2 * gx[addrx+i_1-j_1];
                                            if (ix > 0) { fx += ix * gx[addrx-i_1-j_1]; }
                                            f3x += jx * fx;
                                        }
                                        double f3y = ai2 * gx[addry+i_1+j_1];
                                        if (iy > 0) { f3y += iy * gx[addry-i_1+j_1]; } f3y *= aj2;
                                        if (jy > 0) {
                                            double fy = ai2 * gx[addry+i_1-j_1];
                                            if (iy > 0) { fy += iy * gx[addry-i_1-j_1]; }
                                            f3y += jy * fy;
                                        }
                                        double f3z = ai2 * gx[addrz+i_1+j_1];
                                        if (iz > 0) { f3z += iz * gx[addrz-i_1+j_1]; } f3z *= aj2;
                                        if (jz > 0) {
                                            double fz = ai2 * gx[addrz+i_1-j_1];
                                            if (iz > 0) { fz += iz * gx[addrz-i_1-j_1]; }
                                            f3z += jz * fz;
                                        }
                                        double Ix = gx[addrx];
                                        double Iy = gx[addry];
                                        double Iz = gx[addrz];
                                        double val  = f3x * Iy * Iz;
                                        val += f3y * Ix * Iz;
                                        val += f3z * Ix * Iy;
                                        vj[n] += val * rho;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // while (img_not_processed > 0)
    _filter_ijk_tasks(rem_task_idx, num_ijk_tasks, ijk_tasks_info, (int *)shared_memory);
    } // while (num_ijk_tasks > 0)
    if (thread_id == 0) {
        ksh0_cell0 += nksh;
    }
    __syncthreads();
}

    int bvk_nbas = envs.nbas * ncells;
    int nao = ao_loc[bvk_nbas];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
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
int PBCcontract_int3c2e_pvp_auxvec(double *out, double *auxvec, PBCIntEnvVars *envs,
                                   uint32_t *pool, ShellTripletTaskInfo *task_pool, int *head,
                                   int shm_size, int npairs, int nbatches_ksh, int nauxbas,
                                   uint32_t *bas_ij_idx, int *ksh_offsets,
                                   int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                                   float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    cudaFuncSetAttribute(contract_int3c2e_pvp_auxvec_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int workers = prop.multiProcessorCount;
    cudaMemset(head, 0, sizeof(int));
    contract_int3c2e_pvp_auxvec_kernel<<<workers, THREADS, shm_size>>>(
            out, auxvec, *envs, pool, task_pool, bas_ij_idx, ksh_offsets,
            img_idx, img_offsets, gout_stride_lookup, nauxbas,
            diffuse_exps, diffuse_coefs, log_cutoff,
            head, npairs, nbatches_ksh);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_pvp_auxvec: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
