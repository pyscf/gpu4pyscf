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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "int3c2e_create_tasks_o1.cuh"

#define LMAX            4
#define LMAX1           (LMAX+1)
#define GOUT_WIDTH      54

// lattice sum over i and j for (ij|k)
__global__ static
void pbc_int3c2e_latsum12_kernel(double *out, PBCIntEnvVars envs, int *pool,
                                 uint32_t *bas_ij_idx, int *ksh_offsets,
                                 int *gout_stride_lookup, int *ao_pair_loc, int naux,
                                 float *diffuse_exps, float *diffuse_coefs,
                                 float *atom_coords, float *atom_aux_exps, float log_cutoff)
{
    int ksh_block_id = blockIdx.y;
    int pair_ij = blockIdx.x;
    int thread_id = threadIdx.x;
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int ish, jsh, li, lj, lk, lij, nroots, nfi, nfij, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nimgs_per_block;
    __shared__ double omega;
    __shared__ double *expi, *expj, *ci, *cj;
    __shared__ double xi, yi, zi, xjxi, yjyi, zjzi;
    if (thread_id == 0) {
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        nksh = ksh1 - ksh0;
        ish = bas_ij / bvk_nbas;
        jsh = bas_ij % bvk_nbas;
        li = bas[ish*BAS_SLOTS+ANG_OF];
        lj = bas[jsh*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        lij = li + lj;
        nroots = ((lij + lk) / 2 + 1) * 2;
        iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        omega = env[PTR_RANGE_OMEGA];
        int nfj = (lj + 1) * (lj + 2) / 2;
        int nfk = (lk + 1) * (lk + 2) / 2;
        nfi = (li + 1) * (li + 2) / 2;
        nfij = nfi * nfj;
        nf = nfij * nfk;

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

    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfk = (lk + 1) * (lk + 2) / 2;
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
        idx_i[thread_id] += (thread_id % 3) * nst_per_block * g_size;
    }
    if (thread_id < nfj * 3) {
        idx_j[thread_id] = lex_xyz_address(lj, thread_id) * stride_j * nst_per_block;
    }
    if (thread_id < nfk * 3) {
        idx_k[thread_id] = lex_xyz_address(lk, thread_id) * stride_k * nst_per_block;
    }
    int *img_pool = pool + get_smid() * QUEUE_DEPTH;

    __shared__ int img_counts;
    if (thread_id == 0) {
        img_counts = 0;
    }
    __syncthreads();
    _filter_ij_images(img_counts, img_pool, envs, bas_ij, ksh0, ksh1, li, lj,
                      bas_ij_idx, diffuse_exps, diffuse_coefs,
                      atom_coords, atom_aux_exps, log_cutoff);
    __syncthreads();
    if (img_counts == 0) {
        return;
    }
    for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
        __syncthreads();
        int ksh = ksh0;
        if (kidx < ksh1) {
            ksh = kidx;
        }
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

        for (int img = img_id; img < img_counts+img_id; img += nimgs_per_block) {
            int img_ij = 0;
            if (img < img_counts) {
                img_ij = img_pool[img];
            }
            int jiL = img_ij / nimgs;
            int iL = img_ij - nimgs * jiL;
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
                if (img >= img_counts || kidx >= ksh1) {
                    cicj = 0;
                }
                if (gout_id == 0) {
                    double xiL = xi + img_coords[iL*3+0];
                    double yiL = yi + img_coords[iL*3+1];
                    double ziL = zi + img_coords[iL*3+2];
                    double xjLxi = xjxi + img_coords[jiL*3+0];
                    double yjLyi = yjyi + img_coords[jiL*3+1];
                    double zjLzi = zjzi + img_coords[jiL*3+2];
                    double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                    double Kab = theta_ij * rr_ij;
                    double fac_ij = exp(-Kab);
                    double xij = xjLxi * aj_aij + xiL;
                    double yij = yjLyi * aj_aij + yiL;
                    double zij = zjLzi * aj_aij + ziL;
                    double xpq = xij - rk[0];
                    double ypq = yij - rk[1];
                    double zpq = zij - rk[2];
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
                            if (img < img_counts && kidx < ksh1) {
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
                        if (img < img_counts && kidx < ksh1) {
#pragma unroll
                            for (int n = 0; n < GOUT_WIDTH; ++n) {
                                int ijk = n*gout_stride+gout_id;
                                if (ijk >= nf) break;
                                int k  = ijk / nfij;
                                int ij = ijk - nfij * k;
                                int j = ij / nfi;
                                int i = ij - nfi * j;
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
            __syncthreads();
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
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
        if (img_id == 0 && kidx < ksh1) {
            // save gout in tensor with shape [pair_ij,nfj,nfi, nfk,nksh]
            int k0 = envs.ao_loc[ksh0] - envs.ao_loc[bvk_nbas] + ksh - ksh0;
            size_t pair_offset = ao_pair_loc[pair_ij];
            double *eri3c = out + pair_offset * naux + k0;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ijk = n*gout_stride+gout_id;
                if (ijk >= nf) break;
                int k  = ijk / nfij;
                int ij = ijk - nfij * k;
                eri3c[ij*naux + k*nksh] = gout[n];
            }
        }
    }
}

__global__ static
void bvk_ovlp_mask_kernel(int8_t *ovlp_mask, PBCIntEnvVars envs,
                          float *exps, float *log_coef, float log_cutoff)
{
    int nbas = envs.bvk_ncells * envs.nbas;
    int jsh = blockIdx.x * blockDim.x + threadIdx.x;
    int ish = blockIdx.y * blockDim.y + threadIdx.y;
    if (ish >= nbas || jsh >= nbas) {
        return;
    }
    if (ish < jsh) {
        return;
    }
    int ish_cell0 = ish % envs.nbas;
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
            ovlp_mask[ish*nbas+jsh] = 1;
            ovlp_mask[jsh*nbas+ish] = 1;
            break;
        }
    }
}

extern "C" {
int PBCsr_int3c2e_latsum12(double *out, PBCIntEnvVars *envs, int *pool,
                           int shm_size, int nshl_pair, int nbatches_ksh,
                           uint32_t *bas_ij_idx, int *ksh_offsets,
                           int *gout_stride_lookup, int *ao_pair_loc, int naux,
                           float *diffuse_exps, float *diffuse_coefs,
                           float *atom_coords, float *atom_aux_exps, float log_cutoff)
{
    cudaFuncSetAttribute(pbc_int3c2e_latsum12_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nshl_pair, nbatches_ksh);
    pbc_int3c2e_latsum12_kernel<<<blocks, THREADS, shm_size>>>(
            out, *envs, pool, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, naux,
            diffuse_exps, diffuse_coefs, atom_coords, atom_aux_exps, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_int3c2e: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int bvk_ovlp_mask(int8_t *ovlp_mask, PBCIntEnvVars *envs,
                  float *exps, float *log_coef, float log_cutoff)
{
    int nbas = envs->nbas * envs->bvk_ncells;
    int nbatches = (nbas + 15) / 16;
    dim3 threads(16, 16);
    dim3 blocks(nbatches, nbatches);
    bvk_ovlp_mask_kernel<<<blocks, threads>>>(
            ovlp_mask, *envs, exps, log_coef, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bvk_ovlp_mask: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
