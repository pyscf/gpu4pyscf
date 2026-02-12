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
#include "pbc.cuh"
#include "int3c2e.cuh"
#include "int3c2e_create_tasks.cuh"

#define LMAX            4
#define LMAX1           (LMAX+1)

__global__ static
void int3c2e_ejk_ip1_kernel(double *ejk, double *dm, double *density_auxvec,
                            PBCIntEnvVars envs, ImgIdxPage *page_pool,
                            int *ksh_offsets, int *ksh_idx, uint32_t *bas_ij_idx,
                            int *img_idx, uint32_t *img_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int *batch_aux_offsets, int naux,
                            float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    int ksh_block_id = blockIdx.y;
    int pair_ij = blockIdx.x;
    int thread_id = threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int ncells = envs.bvk_ncells;
    int nbas = envs.cell0_nbas * ncells;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nao = ao_loc[envs.cell0_nbas];
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    __shared__ int kidx0, kidx1, nksh;
    __shared__ int ish, jsh, li, lj, lij, nroots, i0, j0;
    __shared__ int lk, kprim;
    if (thread_id == 0) {
        kidx0 = ksh_offsets[ksh_block_id];
        kidx1 = ksh_offsets[ksh_block_id+1];
        nksh = kidx1 - kidx0;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ksh = ksh_idx[kidx0];
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh*BAS_SLOTS+ANG_OF];
        lij = li + lj + 1;
        nroots = ((lij + lk) / 2 + 1) * 2;
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        i0 = ao_loc[ish];
        j0 = ao_loc[jsh];
    }
    __syncthreads();
    int gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;

    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfk = (lk + 1) * (lk + 2) / 2;
    int nfij = nfi * nfj;
    int nf = nfij * nfk;
    int stride_j = li + 2;
    int stride_k = stride_j * (lj + 1);
    int g_size = stride_k * (lk + 2);
    int i_1 =          nsp_per_block;
    int j_1 = stride_j*nsp_per_block;
    int k_1 = stride_k*nsp_per_block;
    int gx_len = g_size * nsp_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 3 + sp_id;
    double *gx = shared_memory + nsp_per_block * 7 + sp_id;
    double *rw = shared_memory + nsp_per_block * (g_size*3+7) + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    int *idx_k = _c_cartesian_lexical_xyz + lex_xyz_offset(lk);
    page_pool += get_smid() * PAGES_PER_BLOCK;

    double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
    double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
    double ai2 = ai * 2;
    double aj2 = aj * 2;
    __shared__ double aij, aj_aij, theta_ij, cicj;
    __shared__ double xi, yi, zi, xj, yj, zj;
    __shared__ double omega;
    if (thread_id == 0) {
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        xi = ri[0];
        yi = ri[1];
        zi = ri[2];
        xj = rj[0];
        yj = rj[1];
        zj = rj[2];
        aij = ai + aj;
        aj_aij = aj / aij;
        theta_ij = ai * aj_aij;
        cicj = PI_FAC * ci * cj;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.cell0_nbas;
        if (ish_cell0 == jsh_cell0) {
            cicj *= .5;
        } else if (ish_cell0 < jsh_cell0) {
            cicj = 0;
        }
        omega = env[PTR_RANGE_OMEGA];
    }
    if (gout_id == 0) {
        rjri[0*nsp_per_block] = 0;
        rjri[1*nsp_per_block] = 0;
        rjri[2*nsp_per_block] = 0;
        Rpq[0*nsp_per_block] = 0;
        Rpq[1*nsp_per_block] = 0;
        Rpq[2*nsp_per_block] = 0;
        // Rpq[3] must be initialized. An uninitialized Rpq[3] might be nan,
        // which would cause illegal addresses in the rys_roots function
        Rpq[3*nsp_per_block] = 0;
    }

    double v_ix = 0;
    double v_iy = 0;
    double v_iz = 0;
    double v_jx = 0;
    double v_jy = 0;
    double v_jz = 0;

    __shared__ int num_pages, img_max;

    while (kidx0 < kidx1) {
        __syncthreads();
        if (thread_id == 0) {
            num_pages = 0;
        }
        __syncthreads();
        while (kidx0 < kidx1 && num_pages*2 < PAGES_PER_BLOCK) {
            int kidx = kidx0 + thread_id;
            if (kidx < kidx1) {
                _filter_images(num_pages, page_pool, envs, pair_ij, ksh_idx[kidx],
                               kidx, li, lj, bas_ij_idx, img_idx, img_offsets,
                               diffuse_exps, diffuse_coefs, log_cutoff);
            }
            __syncthreads();
            if (thread_id == 0) {
                kidx0 += THREADS;
            }
            __syncthreads();
        }
        if (num_pages >= PAGES_PER_BLOCK) {
            __trap();
        }

        for (int page_id = sp_id; page_id < num_pages+sp_id; page_id += nsp_per_block) {
            __syncthreads();
            ImgIdxPage *page = page_pool + page_id;
            if (page_id >= num_pages) {
                page = page_pool;
            }
            if (thread_id == 0) {
                img_max = 0;
            }
            __syncthreads();
            {
                int img_counts = page->nimgs;
                for (int offset = warpSize/2; offset > 0; offset /= 2) {
                    img_counts = max(img_counts, __shfl_down_sync(0xffffffff, img_counts, offset));
                }
                if (thread_id % warpSize == 0 && gout_id == 0) {
                    atomicMax(&img_max, img_counts);
                }
            }
            __syncthreads();
            int ksh = ksh_idx[page->k];
            int k0;
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = batch_aux_offsets[ksh_block_id] +
                    page->k - ksh_offsets[ksh_block_id];
                size_t pair_offset = ao_pair_loc[page->pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                k0 = ao_loc[ksh] - ao_loc[nbas];
                dm_tensor = dm + j0 * nao + i0;
            }

            if (gout_id == 0) {
                gx[gx_len] = cicj;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            int img_counts = page->nimgs;
            for (int img = 0; img < img_max; ++img) {
                __syncthreads();
                if (gout_id == 0) {
                    if (page_id < num_pages && img < img_counts) {
                        int img_j = page->img_j[img];
                        int img_k = page->img_k[img];
                        double xjL = img_coords[img_j*3+0];
                        double yjL = img_coords[img_j*3+1];
                        double zjL = img_coords[img_j*3+2];
                        double xjxi = xj + xjL - xi;
                        double yjyi = yj + yjL - yi;
                        double zjzi = zj + zjL - zi;
                        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                        double Kab = theta_ij * rr_ij;
                        double fac_ij = exp(-Kab);
                        double xij = xjxi * aj_aij + xi;
                        double yij = yjyi * aj_aij + yi;
                        double zij = zjzi * aj_aij + zi;
                        double xk = rk[0] + img_coords[img_k*3+0];
                        double yk = rk[1] + img_coords[img_k*3+1];
                        double zk = rk[2] + img_coords[img_k*3+2];
                        double xpq = xij - xk;
                        double ypq = yij - yk;
                        double zpq = zij - zk;
                        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                        rjri[0*nsp_per_block] = xjxi;
                        rjri[1*nsp_per_block] = yjyi;
                        rjri[2*nsp_per_block] = zjzi;
                        Rpq[0*nsp_per_block] = xpq;
                        Rpq[1*nsp_per_block] = ypq;
                        Rpq[2*nsp_per_block] = zpq;
                        Rpq[3*nsp_per_block] = rr;
                        gx[gx_len] = cicj * fac_ij;
                    } else {
                        gx[gx_len] = 0.;
                    }
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = expk[kp];
                    double ak2 = ak * 2;
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = ck[kp] / (aij*ak*sqrt(aij+ak));
                    }
                    rys_roots_rs(nroots, theta, Rpq[3*nsp_per_block], omega,
                                 rw, nsp_per_block, gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nsp_per_block];
                        }
                        double rt = rw[ irys*2   *nsp_per_block];
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
                            double c0x = Rpa - rt_aij * Rpq[n*nsp_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nsp_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nsp_per_block] = s2x;
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
                            double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
                            double cpx = rt_ak * Rpq[_ix*nsp_per_block];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsp_per_block];
                                }
                                _gx[stride_k*nsp_per_block] = s1x;
                            }
                            for (int k = 1; k <= lk; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nsp_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }

                        if (lj > 0) {
                            __syncthreads();
                            if (page_id < num_pages) {
                                int lk3 = (lk+2)*3;
                                for (int m = gout_id; m < lk3; m += gout_stride) {
                                    int k = m / 3;
                                    int _ix = m % 3;
                                    double xjxi = rjri[_ix*nsp_per_block];
                                    double *_gx = gx + (_ix*g_size + k*stride_k) * nsp_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nsp_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nsp_per_block];
                                            _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        if (page_id < num_pages) {
                            for (int n = gout_id; n < nf; n+=gout_stride) {
                                int k  = n / nfij;
                                int ij = n % nfij;
                                int j = ij / nfi;
                                int i = ij % nfi;
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
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nsp_per_block;
                                int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nsp_per_block;
                                int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nsp_per_block;
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
                                double fjx = aj2 * (gix - rjri[0*nsp_per_block] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                                double fjy = aj2 * (giy - rjri[1*nsp_per_block] * Iy); if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                                double fjz = aj2 * (giz - rjri[2*nsp_per_block] * Iz); if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                            }
                        }
                    }
                }
            }
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
            double *reduce = shared_memory + thread_id;
            __syncthreads();
            if (page_id < num_pages) {
                reduce[0*threads] = v_kx * 2;
                reduce[1*threads] = v_ky * 2;
                reduce[2*threads] = v_kz * 2;
            }
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i && page_id < num_pages) {
#pragma unroll
                    for (int n = 0; n < 3; ++n) {
                        reduce[n*threads] += reduce[n*threads+i*nsp_per_block];
                    }
                }
            }
            if (gout_id == 0 && page_id < num_pages) {
                atomicAdd(ejk+ka*3+0, reduce[0*threads]);
                atomicAdd(ejk+ka*3+1, reduce[1*threads]);
                atomicAdd(ejk+ka*3+2, reduce[2*threads]);
            }
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
int PBCsr_int3c2e_ejk_ip1(double *ejk, double *dm, double *density_auxvec,
                          PBCIntEnvVars *envs, ImgIdxPage *pool, int shm_size,
                          int npairs, int nbatches_ksh, int *ksh_offsets, int *ksh_idx,
                          uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                          int *gout_stride_lookup,
                          int *ao_pair_loc, int *batch_aux_offsets, int naux,
                          float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    cudaFuncSetAttribute(int3c2e_ejk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(npairs, nbatches_ksh);
    int3c2e_ejk_ip1_kernel<<<blocks, THREADS, shm_size>>>(
            ejk, dm, density_auxvec, *envs, pool, ksh_offsets, ksh_idx,
            bas_ij_idx, img_idx, img_offsets, gout_stride_lookup,
            ao_pair_loc, batch_aux_offsets, naux,
            diffuse_exps, diffuse_coefs, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e_ip1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
