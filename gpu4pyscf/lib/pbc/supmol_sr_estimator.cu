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

#include "gint/cuda_alloc.cuh"
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "gvhf-rys/rys_contract_k.cuh"

#define THREADS         256
#define GOUT_WIDTH      29
#define REMOTE_THRESHOLD 50
// ~= sqrt(-log(1e-16))
#define R_GUESS_FAC     6.f
// float32 underflow limit ~ 3.4e-38. scale by exp(30) to reduce
// rounding errors.
#define UNDERFLOW_GUARD 30.f
#define NEGLIGIBLE_VAL  -700.f
#define SP_BLOCK_SIZE   16384
#define NBAS_MAX        1048576

__global__ static
void fill_s_estimator(float *s_estimator, RysIntEnvVars envs,
                      int64_t *bas_ij_idx, int *bas_mask_idx, float *atom_diffuse_exps,
                      float log_cutoff, int nbas_cell0, int natm_cell0, int npairs,
                      double omega)
{
    int sp_block_id = blockIdx.x;
    int t_id = threadIdx.x;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    int shl_pair0 = sp_block_id * SP_BLOCK_SIZE;
    int shl_pair1 = min((sp_block_id+1) * SP_BLOCK_SIZE, npairs);
    int64_t bas_ij0 = bas_ij_idx[shl_pair0];
    int ish0 = bas_ij0 / NBAS_MAX;
    int jsh0 = bas_ij0 % NBAS_MAX;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    extern __shared__ float shared_memory[];
    float *xyz_cache = shared_memory;
    for (int k = t_id; k < natm_cell0; k += THREADS) {
        double *rk = env + atm[k*ATM_SLOTS+PTR_COORD];
        xyz_cache[k*3+0] = rk[0];
        xyz_cache[k*3+1] = rk[1];
        xyz_cache[k*3+2] = rk[2];
    }
    __syncthreads();

    float omega2 = omega * omega;
    for (int pair_ij = shl_pair0+t_id; pair_ij < shl_pair1; pair_ij += THREADS) {
        int64_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / NBAS_MAX;
        int jsh = bas_ij % NBAS_MAX;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        int ish_cell0 = _ish % nbas_cell0;
        int jsh_cell0 = _jsh % nbas_cell0;
        if (ish_cell0 < jsh_cell0) {
            s_estimator[pair_ij] = NEGLIGIBLE_VAL;
            continue;
        }

        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        float xi = env[ri+0];
        float yi = env[ri+1];
        float zi = env[ri+2];
        float xjxi = env[rj+0] - xi;
        float yjyi = env[rj+1] - yi;
        float zjzi = env[rj+2] - zi;
        float rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        float s_estimator_max = -700.f;
        float ai_cached, aj_cached;
        for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            float ai = env[expi+ip];
            float aj = env[expj+jp];
            float aij = ai + aj;
            float aj_aij = aj / aij;
            float theta_ij = ai * aj / aij;
            float _ci = env[ci+ip];
            float _cj = env[cj+jp];
            float cicj = _ci * _cj;
            float ai_aij = ai / aij;
            float omega2 = omega * omega;
            float fac_guess = .5f - logf(omega2)/4;
            float omega_aij = omega2/(omega2+aij);
            float r_guess = R_GUESS_FAC / sqrtf(aij * omega_aij);
            // log(ci*cj * ((2*li+1)*(2*lj+1))**.5/(4*pi) * (pi/aij)**1.5)
            float norm = 1;
            // s and p functions have been normalized in env[PTR_COEFF].
            // Normalization are applied to d,f,... functions.
            if (li >= 2) { norm *= (2*li+1.f) / (4*M_PI); }
            if (lj >= 2) { norm *= (2*lj+1.f) / (4*M_PI); }
            float log_fac = logf(fabsf(cicj)*sqrtf(norm)) + 1.7171f - 1.5f*logf(aij) + fac_guess;
            float dri = aj_aij * r_guess;
            float drj = ai_aij * r_guess;
            float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
            float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
            float estimator = dri_fac + drj_fac - theta_ij*rr_ij + log_fac;
            if (estimator > s_estimator_max) {
                ai_cached = ai;
                aj_cached = aj;
                s_estimator_max = estimator;
            }
        }

        if (s_estimator_max > NEGLIGIBLE_VAL) {
            float aij = ai_cached + aj_cached;
            float aj_aij = aj_cached / aij;
            float xpa = xjxi * aj_aij;
            float ypa = yjyi * aj_aij;
            float zpa = zjzi * aj_aij;
            float xij = xi + xpa;
            float yij = yi + ypa;
            float zij = zi + zpa;
            float theta = omega2 * aij / (omega2 + aij);
            float s_ij = s_estimator_max;
            float rr_cutoff = s_ij - log_cutoff;  
            int negligible = 1;
            for (int k = 0; k < natm_cell0; ++k) {
                float dx = xij - xyz_cache[k*3+0];
                float dy = yij - xyz_cache[k*3+1];
                float dz = zij - xyz_cache[k*3+2];
                float rr = dx * dx + dy * dy + dz * dz;
                // The density distribution exp(-ak) gives the upper bound of
                // the orbital pair products almost everywhere. Use exp(-ak) to
                // approximate the orbital pair.
                float ak = atom_diffuse_exps[k];
                int lk = 1;
                float omega2_ak = omega2 / (omega2 + ak);
                float log_rt_ak = max(0.f, logf(omega2_ak*omega2*rr + lk/(2*ak)));
                float rr_cutoff_w_penalty = rr_cutoff
                    + 1.7171f - 1.5f*logf(ak) + .5f*lk*log_rt_ak;
                float theta_k = theta * ak / (theta + ak);
                //float theta_rr = logf(rr + 1.f) + theta_k * rr;
                float theta_rr = theta_k * rr;
                if (theta_rr < rr_cutoff_w_penalty) {
                    negligible = 0;
                    break;
                }
            }
            if (negligible) {
                s_estimator_max = NEGLIGIBLE_VAL;
            }
        }
        s_estimator[pair_ij] = s_estimator_max;
    }
    __syncthreads();
}

__global__ static
void q_cond_kernel(float *q_cond, RysIntEnvVars envs,
                   int64_t *bas_ij_idx, int *gout_stride_lookup,
                   int npairs, double omega)
{
    int sp_block_id = blockIdx.x;
    int threads = blockDim.x;
    int t_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    int shl_pair0 = sp_block_id * SP_BLOCK_SIZE;
    int shl_pair1 = min((sp_block_id+1) * SP_BLOCK_SIZE, npairs);
    int64_t bas_ij0 = bas_ij_idx[shl_pair0];
    int ish0 = bas_ij0 / NBAS_MAX;
    int jsh0 = bas_ij0 % NBAS_MAX;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = iprim;
    int lprim = jprim;
    int lij = li + lj;
    int nroots = (lij + 1) * 2;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int nfij = nfi * nfj;

    __shared__ int gout_stride, nsp_per_block;
    if (t_id == 0) {
        gout_stride = gout_stride_lookup[li*LMAX1+lj];
        nsp_per_block = THREADS / gout_stride;
    }
    __syncthreads();
    int sp_id = t_id % nsp_per_block;
    int gout_id = t_id / nsp_per_block;

    int g_size = stride_k;
    extern __shared__ float shared_memory[];
    float *rjri = shared_memory + sp_id;
    float *Rpq = shared_memory + nsp_per_block * 3 + sp_id;
    float *rw = shared_memory + nsp_per_block * 6 + sp_id;
    // Generate the double-precision quadratures in rw_cache then copy to rw
    double *rw_cache = (double *)(shared_memory + nsp_per_block * 6 + threads) + sp_id;
    float *gx = shared_memory + nsp_per_block * (nroots * 2 + 6) + sp_id;
    // gz can be reused for gbuf; gbuf size = (li+1)*(lj+1)*(lij+1)
    float *gbuf = gx + g_size * nsp_per_block * 2;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);

    for (int task_id = shl_pair0+sp_id; task_id < shl_pair1+sp_id; task_id += nsp_per_block) {
        float gout[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            gout[n] = 0.;
        }
        int pair_ij = task_id;
        if (pair_ij >= shl_pair1) {
            pair_ij = shl_pair0;
        }
        int64_t bas_ij = bas_ij_idx[pair_ij];
        int64_t ish = bas_ij / NBAS_MAX;
        int64_t jsh = bas_ij % NBAS_MAX;
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int expk = expi;
        int expl = expj;
        int ck = ci;
        int cl = cj;
        float xjxi = env[rj+0] - env[ri+0];
        float yjyi = env[rj+1] - env[ri+1];
        float zjzi = env[rj+2] - env[ri+2];
        if (gout_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        float rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        float rr_kl = rr_ij;

        for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
            __syncthreads();
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            float ai = env[expi+ip];
            float aj = env[expj+jp];
            float aij = ai + aj;
            float aj_aij = aj / aij;
            float theta_ij = ai * aj / aij;
            float _ci = env[ci+ip];
            float _cj = env[cj+jp];
            float cicj = _ci * _cj;
            float Kab = expf(UNDERFLOW_GUARD - theta_ij * rr_ij);
            cicj *= Kab / aij * PI_FAC;

            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                int kp = klp / lprim;
                int lp = klp % lprim;
                float ak = env[expk+kp];
                float al = env[expl+lp];
                float akl = ak + al;
                float al_akl = al / akl;
                float theta_kl = ak * al / akl;
                float Kcd = expf(UNDERFLOW_GUARD - theta_kl * rr_kl);
                float _ck = env[ck+kp];
                float _cl = env[cl+lp];
                float ckcl = _ck * _cl * Kcd / akl;
                float fac = cicj * ckcl / sqrtf(aij+akl);
                float xij = env[ri+0] + rjri[0*nsp_per_block] * aj_aij;
                float yij = env[ri+1] + rjri[1*nsp_per_block] * aj_aij;
                float zij = env[ri+2] + rjri[2*nsp_per_block] * aj_aij;
                float xkl = env[ri+0] + rjri[0*nsp_per_block] * al_akl;
                float ykl = env[ri+1] + rjri[1*nsp_per_block] * al_akl;
                float zkl = env[ri+2] + rjri[2*nsp_per_block] * al_akl;
                float xpq = xij - xkl;
                float ypq = yij - ykl;
                float zpq = zij - zkl;
                if (gout_id == 0) {
                    Rpq[0*nsp_per_block] = xpq;
                    Rpq[1*nsp_per_block] = ypq;
                    Rpq[2*nsp_per_block] = zpq;
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij / (aij + akl) * akl;
                double lr_factor = 0.;
                double sr_factor = 1.;
                rys_roots_for_k(nroots, theta, rr, rw_cache, omega, lr_factor, sr_factor,
                                nsp_per_block, gout_stride, gout_id);
                for (int n = 0; n < nroots*2; ++n) {
                    __syncthreads();
                    if (gout_id == 0) {
                        rw[n*nsp_per_block] = rw_cache[n*nsp_per_block];
                    }
                }
                for (int irys = nroots-1; irys >= 0; --irys) {
                    __syncthreads();
                    if (lij > 0 && gout_id == 0) {
                        float rt = rw[irys*2*nsp_per_block];
                        float rt_aa = rt / (aij + akl);
                        float rt_aij = rt_aa * akl;
                        float rt_akl = rt_aa * aij;
                        float b00 = .5 * rt_aa;
                        float b10 = .5/aij * (1 - rt_aij);
                        float b01 = .5/akl * (1 - rt_akl);
                        float s0x, s1x, s2x;
                        for (int _ix = 0; _ix < 3; _ix++) {
                            float xjxi = rjri[_ix*nsp_per_block];
                            float xlxk = xjxi;
                            float Rpa = xjxi * aj_aij;
                            float c0x = Rpa - rt_aij * Rpq[_ix*nsp_per_block];
                            float Rqc = xlxk * al_akl;
                            float cpx = Rqc + rt_akl * Rpq[_ix*nsp_per_block];
                            if (_ix == 2) {
                                gbuf[0] = fac * rw[(irys*2+1)*nsp_per_block];
                            } else {
                                gbuf[0] = 1;
                            }
                            // TRR
                            s0x = gbuf[0];
                            s1x = c0x * s0x;
                            gbuf[nsp_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                gbuf[(i+1)*nsp_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                            for (int i = 0; i <= lij; i++) {
                                float *_gx = gbuf + i * nsp_per_block;
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsp_per_block];
                                }
                                _gx[stride_k*nsp_per_block] = s1x;
                                for (int k = 1; k < lij; ++k) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nsp_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nsp_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }

                            // hrr
                            if (lj > 0) {
                                for (int k = 0; k <= lij; k++) {
                                    float *_gx = gbuf + k*stride_k * nsp_per_block;
                                    for (int j = 0; j < lj; ++j) {
                                        int ij = lij + j*li; // = (lij-j) + j*stride_j;
                                        s1x = _gx[ij*nsp_per_block];
                                        for (--ij; ij >= j*stride_j; --ij) {
                                            s0x = _gx[ij*nsp_per_block];
                                            _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }

                            float *gz = gx + _ix * g_size * nsp_per_block;
                            for (int i = 0; i <= li; i++) {
                                gz[i*nsp_per_block] = gbuf[(i+i*stride_k)*nsp_per_block];
                            }
                            for (int jj = 1; jj <= lj; jj++) {
                                for (int ij = jj*stride_j; ij < stride_k; ij++) {
                                    // ij = i+j*stride_j
                                    float *_gx = gbuf + ij * nsp_per_block;
                                    s0x = _gx[0];
                                    for (int k = 0; k <= lij-jj; k++) {
                                        s1x = _gx[(k+1)*stride_k*nsp_per_block];
                                        _gx[k*stride_k*nsp_per_block] = s1x - xlxk * s0x;
                                        s0x = s1x;
                                    }
                                }
                                float *gz = gx + (_ix * g_size + jj*stride_j) * nsp_per_block;
                                for (int i = 0; i <= li; i++) {
                                    gz[i*nsp_per_block] = gbuf[(i+jj*stride_j+i*stride_k)*nsp_per_block];
                                }
                            }
                        }
                    } else {
                        gx[0] = fac;
                        gx[g_size*nsp_per_block] = 1;
                        gx[g_size*nsp_per_block*2] = rw[(irys*2+1)*nsp_per_block];
                    }

                    __syncthreads();
                    if (task_id >= shl_pair1) {
                        continue;
                    }
                    float div_nfi = c_div_nf[li];
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        int ij = n*gout_stride + gout_id;
                        if (ij >= nfij) break;
                        int j = ij * div_nfi;
                        int i = ij - nfi * j;
                        int ix = idx_i[i*3+0];
                        int iy = idx_i[i*3+1];
                        int iz = idx_i[i*3+2];
                        int jx = idx_j[j*3+0];
                        int jy = idx_j[j*3+1];
                        int jz = idx_j[j*3+2];
                        int addrx = (ix + jx*stride_j) * nsp_per_block;
                        int addry = (iy + jy*stride_j + g_size) * nsp_per_block;
                        int addrz = (iz + jz*stride_j + g_size*2) * nsp_per_block;
                        gout[n] += gx[addrx] * gx[addry] * gx[addrz];
                    }
                }
            }
        }
        float gout_max = 0;
        if (task_id < shl_pair1) {
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride+gout_id;
                if (ij >= nfij) break;
                gout_max = max(fabsf(gout[n]), gout_max);
            }
        }
        float *reduce = shared_memory + t_id;
        reduce[0] = gout_max;
        __syncthreads();
        if (gout_id == 0 && task_id < shl_pair1) {
            for (int i = 1; i < gout_stride; ++i) {
                gout_max = max(gout_max, reduce[i*nsp_per_block]);
            }
            float log_q;
            if (gout_max == 0) {
                log_q = -700.f;
            } else {
                log_q = logf(gout_max) / 2 - UNDERFLOW_GUARD;
            }
            q_cond[pair_ij] = log_q;
        }
    }
}

// Perform
//   ish = ish.reshape(-1, tile)
//   jsh = jsh.reshape(-1, tile)
//   pair_ij = ish[:,None,:,None] * NBAS_MAX + jsh[None,:,None,:]
__global__ static
void sort_pair_ij_kernel(int64_t *pair_ij, int *ish, int *jsh, int nish, int njsh, int tile)
{
    int t_id = threadIdx.x;
    int threads = blockDim.x;
    int i_tile = blockIdx.x;
    size_t off = i_tile * tile * (size_t)njsh;
    // when nish not divisible by tile
    int nish_rem = min(tile, nish - i_tile * tile);
    float div_tile = 1.f / tile;
    float div_tile2 = div_tile / nish_rem;
    int tile2 = nish_rem * tile;
    for (int n = t_id; n < nish_rem*njsh; n += threads) {
        int j_tile = n * div_tile2;
        int ijr = n - j_tile * tile2;
        int ir = ijr * div_tile;
        int jr = ijr - ir * tile;
        int njsh_rem = njsh - j_tile * tile;
        if (njsh_rem < tile) { // when njsh not divisible by tile
            ir = ijr / njsh_rem;
            jr = ijr - njsh_rem * ir;
        }
        int i = i_tile * tile + ir;
        int j = j_tile * tile + jr;
        if (i >= nish) break;
        pair_ij[off+n] = (int64_t)ish[i] * NBAS_MAX + jsh[j];
    }
}

extern "C" {
int PBCfill_s_estimator(float *s_estimator, RysIntEnvVars *envs,
                        int64_t *bas_ij_idx, int *bas_mask_idx, float *atom_diffuse_exps,
                        float log_cutoff, int nbas_cell0, int natm_cell0, int npairs,
                        double omega)
{
    int sp_blocks = (npairs + SP_BLOCK_SIZE - 1) / SP_BLOCK_SIZE;
    int buflen = max(512, natm_cell0 * 3) * sizeof(float);
    fill_s_estimator<<<sp_blocks, THREADS, buflen>>>(
        s_estimator, *envs, bas_ij_idx, bas_mask_idx, atom_diffuse_exps,
        log_cutoff, nbas_cell0, natm_cell0, npairs, omega);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBCfill_s_estimator %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCfill_qcond(float *q_cond, RysIntEnvVars *envs, int shm_size,
                  int64_t *bas_ij_idx, int *gout_stride_lookup,
                  int npairs, double omega)
{
    int sp_blocks = (npairs + SP_BLOCK_SIZE - 1) / SP_BLOCK_SIZE;
    q_cond_kernel<<<sp_blocks, THREADS, shm_size>>>(
        q_cond, *envs, bas_ij_idx, gout_stride_lookup, npairs, omega);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBCfill_qcond %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBCsort_pair_ij(int64_t *pair_ij, int *ish, int *jsh, int nish, int njsh, int tile)
{
    int ntile = (nish + tile - 1) / tile;
    sort_pair_ij_kernel<<<ntile, THREADS>>>(pair_ij, ish, jsh, nish, njsh, tile);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBCsort_pair_ij %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
