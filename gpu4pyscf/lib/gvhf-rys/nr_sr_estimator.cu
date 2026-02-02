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

#include "gint/cuda_alloc.cuh"
#include "vhf.cuh"
#include "rys_roots_for_k.cu"
//#include "create_tasks.cu"
#include "rys_contract_k.cuh"

#define THREADS         256
#define GOUT_WIDTH      60
#define REMOTE_THRESHOLD 50
// sqrt(-log(1e-9))
#define R_GUESS_FAC     4.5f

static __global__
void int2e_qcond_kernel(float *q_out, float *s_out, RysIntEnvVars envs,
                        int *shl_pair_offsets, uint32_t *bas_ij_idx, int *gout_stride_lookup,
                        double omega, double lr_factor, double sr_factor
                        #ifdef USE_SYCL
                        , sycl::nd_item<1> &item, char* shm_size
                        #endif
                        )
{
    #ifdef USE_SYCL
    int sp_block_id = item.get_group(0);
    int thread_id = item.get_local_id(0);    
    float* shared_memory = reinterpret_cast<float*>(shm_size);
    #else
    int sp_block_id = blockIdx.x;
    int thread_id = threadIdx.x;    
    extern __shared__ float shared_memory[];  
    #endif
    int shl_pair0 = shl_pair_offsets[sp_block_id];
    int shl_pair1 = shl_pair_offsets[sp_block_id+1];
    int bas_ij0 = bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int *bas = envs.bas;
    double *env = envs.env;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    if (li > LMAX || lj > LMAX) {
        return;
    }
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = iprim;
    int lprim = jprim;
    int lij = li + lj;
    int nroots = lij + 1;
    if (omega < 0) {
        nroots *= 2;
    }
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int nfij = nfi * nfj;

    int gout_stride = gout_stride_lookup[li*LMAX1+lj];
    int nsp_per_block = THREADS / gout_stride;
    int sp_id = thread_id % nsp_per_block;
    int gout_id = thread_id / nsp_per_block;

    int g_size = stride_k;
    double *rw = ((double *)shared_memory) + sp_id;
    float *rjri = shared_memory + nsp_per_block * nroots * 2 * 2 + sp_id;
    float *Rpq = shared_memory + nsp_per_block * (nroots * 4 + 3) + sp_id;
    float *gx = shared_memory + nsp_per_block * (nroots * 4 + 6) + sp_id;
    // gz can be reused for gbuf
    float *gbuf = gx + g_size * nsp_per_block * 2;
    const int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    const int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);
    gx[0] = 1;
    gx[g_size*nsp_per_block] = 1;

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
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *expk = expi;
        double *expl = expj;
        double *ck = ci;
        double *cl = cj;
        float xjxi = rj[0] - ri[0];
        float yjyi = rj[1] - ri[1];
        float zjzi = rj[2] - ri[2];
        if (gout_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        float rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        float rr_kl = rr_ij;

        float s_estimator_max = -700.f;
        for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
            __syncthreads();
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            float ai = expi[ip];
            float aj = expj[jp];
            float aij = ai + aj;
            float aj_aij = aj / aij;
            float theta_ij = ai * aj / aij;
            float cicj = ci[ip] * cj[jp];
            if (s_out != NULL && omega != 0 && gout_id == 0 && task_id < shl_pair1) {
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
                s_estimator_max = max(s_estimator_max, estimator);
            }
            if (q_out == NULL) {
                continue;
            }

            // float32 underflow limit ~ 3.4e-38. scale by exp(30) to reduce
            // rounding errors.
            float Kab = expf(30.f - theta_ij * rr_ij);
            cicj *= Kab / aij * PI_FAC;

            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                int kp = klp / lprim;
                int lp = klp % lprim;
                float ak = expk[kp];
                float al = expl[lp];
                float akl = ak + al;
                float al_akl = al / akl;
                float theta_kl = ak * al / akl;
                float Kcd = expf(30.f - theta_kl * rr_kl);
                float ckcl = ck[kp] * cl[lp] * Kcd / akl;
                float fac = cicj * ckcl / sqrtf(aij+akl);
                float xij = ri[0] + rjri[0*nsp_per_block] * aj_aij;
                float yij = ri[1] + rjri[1*nsp_per_block] * aj_aij;
                float zij = ri[2] + rjri[2*nsp_per_block] * aj_aij;
                float xkl = ri[0] + rjri[0*nsp_per_block] * al_akl;
                float ykl = ri[1] + rjri[1*nsp_per_block] * al_akl;
                float zkl = ri[2] + rjri[2*nsp_per_block] * al_akl;
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
                rys_roots_for_k(nroots, theta, rr, rw, omega, lr_factor, sr_factor,
                                nsp_per_block, gout_stride, gout_id);
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
                        gx[nsp_per_block*g_size*2] = fac * rw[(irys*2+1)*nsp_per_block];
                    }

                    __syncthreads();
                    if (task_id >= shl_pair1) {
                        continue;
                    }
                    float div_nfi = c_div_nf[li];
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        uint32_t ij = n*gout_stride + gout_id;
                        if (ij >= nfij) break;
                        uint32_t j = ij * div_nfi;
                        uint32_t i = ij - nfi * j;
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
        if (q_out != NULL) {
            float gout_max = 0;
            if (task_id < shl_pair1) {
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ij = n*gout_stride+gout_id;
                    if (ij >= nfij) break;
                    gout_max = max(fabsf(gout[n]), gout_max);
                }
            }
            float *reduce = shared_memory + thread_id;
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
                    log_q = logf(gout_max) / 2 - 30.f;
                }
                q_out[ish*nbas+jsh] = log_q;
                q_out[jsh*nbas+ish] = log_q;
            }
        }
        if (s_out != NULL && gout_id == 0 && task_id < shl_pair1) {
            s_out[ish*nbas+jsh] = s_estimator_max;
            s_out[jsh*nbas+ish] = s_estimator_max;
        }
    }
}

extern "C" {
int int2e_qcond_estimator(float *q_out, float *s_out, RysIntEnvVars *envs, int shm_size,
                          int nbatches_shl_pair, uint32_t *bas_ij_idx,
                          int *shl_pair_offsets, int *gout_stride_lookup,
                          double omega, double lr_factor, double sr_factor)
{
    cudaFuncSetAttribute(int2e_qcond_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    #ifdef USE_SYCL
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class int2e_qcond_sycl>(sycl::nd_range<1>(nbatches_shl_pair * THREADS, THREADS), [=](auto item) {
        int2e_qcond_kernel(q_out, s_out, dev_envs, shl_pair_offsets, bas_ij_idx,
                           gout_stride_lookup, omega, lr_factor, sr_factor,
                           item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    int2e_qcond_kernel<<<nbatches_shl_pair, THREADS, shm_size>>>(
            q_out, s_out, *envs, shl_pair_offsets, bas_ij_idx,
            gout_stride_lookup, omega, lr_factor, sr_factor);
    #endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in int1e_ovlp kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
