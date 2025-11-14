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

#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/boys.cu"
#include "gvhf-md/md_j.cuh"

#define RT2_MAX 9
#define IJ_SIZE 11
// 48KB ~18, 96KB ~41, 160KB ~61
#define RT_TMP_SIZE 31
#define RT2_IDX_CACHE_SIZE (35*56)

extern __constant__ uint16_t c_Rt_idx[];
extern __constant__ int8_t c_Rt_tuv_fac[];
extern __constant__ int8_t c_Rt2_efg_phase[];
extern __device__ int Rt2_idx_offsets[];
extern __device__ uint16_t Rt2_kl_ij[];
extern __device__ uint16_t Rt2_ij_kl[];

#define ADDR(l, t, u, v) \
        ((l+1)*(l+2)*(l+3)/6 - ((l)-(t)+1)*((l)-(t)+2)*((l)-(t)+3)/6 + \
         ((l)-(t)+1)*((l)-(t)+2)/2 - ((l)-(t)-(u)+1)*((l)-(t)-(u)+2)/2 + (v))

__device__
inline void iter_Rt_n(double *Rt, double rx, double ry, double rz, int l,
                      int nsq_per_block, int gout_id, int gout_stride)
{
    int nf2 = (l + 1) * (l + 2) / 2;
    int nf3 = nf2 * (l + 3) / 3;
    int offsets = nf3 * l / 4 - l; //l*(l+1)*(l+2)*(l+3)/24 - l;
    uint16_t *p1 = c_Rt_idx + offsets;
    int8_t *tuv_fac = c_Rt_tuv_fac + offsets;
    double Rt_tmp[RT_TMP_SIZE];
    nf2 -= 1; // Drop the first element in Rt. It is assigned outside
    nf3 -= 1;
    for (int n = 0; n < RT_TMP_SIZE; ++n) {
        int i = n * gout_stride + gout_id;
        if (i >= nf3) break;
        Rt_tmp[n] = tuv_fac[i] * Rt[p1[i]*nsq_per_block];
        if (i < l) {
            Rt_tmp[n] += rz * Rt[i*nsq_per_block];
        } else if (i < nf2) {
            Rt_tmp[n] += ry * Rt[(i-l)*nsq_per_block];
        } else {
            Rt_tmp[n] += rx * Rt[(i-nf2)*nsq_per_block];
        }
    }
    __syncthreads();
    for (int n = 0; n < RT_TMP_SIZE; ++n) {
        int i = n * gout_stride + gout_id;
        if (i >= nf3) break;
        Rt[(i+1)*nsq_per_block] = Rt_tmp[n];
    }
}

// gout_pattern = ((li == 0) >> 3) | ((lj == 0) >> 2) | ((lk == 0) >> 1) | (ll == 0);
__global__
void pbc_md_j_kernel(RysIntEnvVars envs, JKMatrix jmat, MDBoundsInfo bounds,
                  int threadsx, int threadsy, int tilex, int tiley,
                  uint16_t *pRt2_kl_ij, int8_t *efg_phase)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int bsizex = threadsx * tilex;
    int bsizey = threadsy * tiley;
    int task_ij0 = blockIdx.x * bsizex;
    int task_kl0 = blockIdx.y * bsizey;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping &&
        // when ij pattern and kl pattern are identical, the 8-fold permutation
        // symmetry can be utilized. Tiles on in the upper triangular part can
        // be skipped. If the last ij task (task_ij0+bsizex-1) is greater than
        // the first kl task (task_kl0), tile is completely inside the triu part.
        task_ij0+bsizex <= task_kl0) {
        return;
    }

    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int nsq_per_block = blockDim.x;
    //assert(nsq_per_block == threadsx * threadsy);
    int t_id = gout_id * nsq_per_block + sq_id;
    int tx = sq_id % threadsx;
    int ty = sq_id / threadsx;
    int threads = nsq_per_block * gout_stride;
    int order = bounds.order;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jmat.dm;
    double *vj = jmat.vj;
    int nf3ij = bounds.nf3ij;
    int nf3kl = bounds.nf3kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double dm_kl_cache[];
    double *Rq_cache = dm_kl_cache + nf3kl*bsizey;
    double *Rp_cache = dm_kl_cache + bsizey*(4+nf3kl);
    double *gamma_inc = dm_kl_cache + bsizey*(4+nf3kl) + threadsx*4 + sq_id;
    double *Rt = gamma_inc + (order+1) * nsq_per_block;
    uint16_t *Rt2_address = pRt2_kl_ij;
    if (nf3ij * nf3kl <= RT2_IDX_CACHE_SIZE) {
        int l4 = bounds.lij + bounds.lkl;
        int nf3 = (l4 + 1) * (l4 + 2) * (l4 + 3) / 6;
        Rt2_address = (uint16_t *)(Rt - sq_id + nf3 * nsq_per_block);
        for (int n = t_id; n < nf3ij * nf3kl; n += threads) {
            Rt2_address[n] = pRt2_kl_ij[n];
        }
    }
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    __syncthreads();
    for (int n = t_id; n < bsizey; n += threads) {
        int task_kl = blockIdx.y * bsizey + n;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            double ak = env[bas[ksh*BAS_SLOTS+PTR_EXP]];
            double al = env[bas[lsh*BAS_SLOTS+PTR_EXP]];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double akl = ak + al;
            double xkl = (ak * rk[0] + al * rl[0]) / akl;
            double ykl = (ak * rk[1] + al * rl[1]) / akl;
            double zkl = (ak * rk[2] + al * rl[2]) / akl;
            Rq_cache[n+0*bsizey] = xkl;
            Rq_cache[n+1*bsizey] = ykl;
            Rq_cache[n+2*bsizey] = zkl;
            Rq_cache[n+3*bsizey] = akl;
        } else {
            Rq_cache[n+0*bsizey] = 1e5;
            Rq_cache[n+1*bsizey] = 1e5;
            Rq_cache[n+2*bsizey] = 1e5;
            Rq_cache[n+3*bsizey] = 1.;
        }
    }
    {
        int xslots = threadsx * gout_stride;
        int xslot_id = t_id / threadsy;
        int ty = t_id % threadsy;
        for (int n = xslot_id; n < nf3kl * tiley; n += xslots) {
            int kl = n / tiley;
            int batch_kl = n  - kl * tiley;
            int sq_kl = ty + batch_kl * threadsy;
            int task_kl = blockIdx.y * bsizey + sq_kl;
            if (task_kl < npairs_kl) {
                int kl_loc0 = pair_kl_loc[task_kl];
                dm_kl_cache[sq_kl+kl*bsizey] = dm[kl_loc0+kl];
            }
        }
    }

    for (int batch_ij = 0; batch_ij < tilex; ++batch_ij) {
        int task_ij0 = (blockIdx.x * tilex + batch_ij) * threadsx;
        if (task_ij0 >= npairs_ij) {
            break;
        }
        __syncthreads();
        if (t_id < threadsx) {
            int task_ij = task_ij0 + t_id;
            if (task_ij < npairs_ij) {
                int pair_ij = pair_ij_mapping[task_ij];
                int ish = pair_ij / nbas;
                int jsh = pair_ij % nbas;
                double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
                double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                double aij = ai + aj;
                double xij = (ai * ri[0] + aj * rj[0]) / aij;
                double yij = (ai * ri[1] + aj * rj[1]) / aij;
                double zij = (ai * ri[2] + aj * rj[2]) / aij;
                Rp_cache[t_id+0*threadsx] = xij;
                Rp_cache[t_id+1*threadsx] = yij;
                Rp_cache[t_id+2*threadsx] = zij;
                Rp_cache[t_id+3*threadsx] = aij;
            } else {
                Rp_cache[t_id+0*threadsx] = 2e5;
                Rp_cache[t_id+1*threadsx] = 2e5;
                Rp_cache[t_id+2*threadsx] = 2e5;
                Rp_cache[t_id+3*threadsx] = 1.; // aij
            }
        }
        double vj_ij[IJ_SIZE];
#pragma unroll
        for (int n = 0; n < IJ_SIZE; ++n) {
            vj_ij[n] = 0.;
        }
        for (int batch_kl = 0; batch_kl < tiley; ++batch_kl) {
            int task_kl0 = (blockIdx.y * tiley + batch_kl) * threadsy;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*tilex+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*tiley+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * threadsy;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
            __syncthreads();
            int bsizey = threadsy * tiley;
            double xij = Rp_cache[tx+0*threadsx];
            double yij = Rp_cache[tx+1*threadsx];
            double zij = Rp_cache[tx+2*threadsx];
            double aij = Rp_cache[tx+3*threadsx];
            double xkl = Rq_cache[sq_kl+0*bsizey];
            double ykl = Rq_cache[sq_kl+1*bsizey];
            double zkl = Rq_cache[sq_kl+2*bsizey];
            double akl = Rq_cache[sq_kl+3*bsizey];
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            if (gout_id == 0) {
                double omega = jmat.omega;
                boys_fn(gamma_inc, theta, rr, omega, fac/(aij*akl*sqrt(aij+akl)),
                        order, 0, nsq_per_block);
                Rt[0] = gamma_inc[order*nsq_per_block];
            }
            for (int n = 1; n <= order; ++n) {
                __syncthreads();
                if (n == 1) {
                    if (gout_id == 0) {
                        double _Rt_0 = Rt[0];
                        Rt[1*nsq_per_block] = zpq * _Rt_0;
                        Rt[2*nsq_per_block] = ypq * _Rt_0;
                        Rt[3*nsq_per_block] = xpq * _Rt_0;
                        Rt[0] = gamma_inc[(order-n)*nsq_per_block];
                    }
                } else if (n == 2) {
                    if (gout_id == 0) {
                        double _Rt_0 = Rt[0];
                        double _Rt_1 = Rt[1*nsq_per_block];
                        double _Rt_2 = Rt[2*nsq_per_block];
                        double _Rt_3 = Rt[3*nsq_per_block];
                        Rt[1*nsq_per_block] = zpq * _Rt_0;
                        Rt[2*nsq_per_block] = zpq * _Rt_1 + _Rt_0;
                        Rt[3*nsq_per_block] = ypq * _Rt_0;
                        Rt[4*nsq_per_block] = ypq * _Rt_1;
                        Rt[5*nsq_per_block] = ypq * _Rt_2 + _Rt_0;
                        Rt[6*nsq_per_block] = xpq * _Rt_0;
                        Rt[7*nsq_per_block] = xpq * _Rt_1;
                        Rt[8*nsq_per_block] = xpq * _Rt_2;
                        Rt[9*nsq_per_block] = xpq * _Rt_3 + _Rt_0;
                        Rt[0] = gamma_inc[(order-n)*nsq_per_block];
                    }
                } else {
                    iter_Rt_n(Rt, xpq, ypq, zpq, n, nsq_per_block, gout_id, gout_stride);
                    if (gout_id == 0) {
                        Rt[0] = gamma_inc[(order-n)*nsq_per_block];
                    }
                }
            }
            __syncthreads();

            if (task_kl < npairs_kl) {
                for (int k = 0; k < nf3kl; ++k) {
                    double dm_kl = efg_phase[k] * dm_kl_cache[k*bsizey+sq_kl];
                    uint16_t *p1_ij = Rt2_address + k * nf3ij;
#pragma unroll
                    for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                        if (i >= nf3ij) break;
                        double s = Rt[p1_ij[i]*nsq_per_block];
                        vj_ij[n] += s * dm_kl;
                    }
                }
            }
        }
        {
            double *vj_cache = Rp_cache + t_id;
            int task_ij = task_ij0 + tx;
            int ij_loc0 = pair_ij_loc[task_ij];
#pragma unroll
            for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                if (i >= nf3ij+gout_id) break;
                __syncthreads();
                vj_cache[0] = vj_ij[n];
                for (int stride = threadsy/2; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache[0] += vj_cache[stride*threadsx];
                    }
                }
                __syncthreads();
                if (ty == 0 && i < nf3ij && task_ij < npairs_ij) {
                    atomicAdd(vj+ij_loc0+i, vj_cache[0]);
                }
            }
        }
    }
}

extern "C" {
int PBC_build_j(double *vj, double *dm, int n_dm,
                int dm_xyz_size, int nimgs_uniq_pair,
                RysIntEnvVars *envs, int *scheme, int *shls_slice,
                int npairs_ij, int npairs_kl,
                int *pair_ij_mapping, int *pair_kl_mapping,
                int *pair_ij_loc, int *pair_kl_loc,
                float *qd_ij_max, float *qd_kl_max,
                float *q_cond, float cutoff,
                int *atm, int natm, int *bas, int nbas, double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4];
    int lsh0 = shls_slice[6];
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    int lij = li + lj;
    int lkl = lk + ll;
    int order = lij + lkl;
    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
    int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
    int nf3ijkl = (order+1)*(order+2)*(order+3)/6;
    // 16x16 threads are applied to all unrolled code
    float *tile16_qd_ij_max = qd_ij_max + qd_offset_for_threads(npairs_ij, 16);
    float *tile16_qd_kl_max = qd_kl_max + qd_offset_for_threads(npairs_kl, 16);
    MDBoundsInfo bounds = {li, lj, lk, ll, lij, lkl, order, nf3ij, nf3kl, nf3ijkl,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        pair_ij_loc, pair_kl_loc, tile16_qd_ij_max, tile16_qd_kl_max,
        q_cond, cutoff};

    double omega = env[PTR_RANGE_OMEGA];
    int threads_ij = scheme[0];
    int threads_kl = scheme[1];
    int gout_stride = scheme[2];
    int tilex = scheme[3];
    int tiley = scheme[4];
    int buflen = scheme[5];
    int bsizex = threads_ij * tilex;
    int bsizey = threads_kl * tiley;
    int nsq_per_block = threads_ij * threads_kl;
    dim3 threads(nsq_per_block, gout_stride);
    int blocks_ij = (npairs_ij + bsizex - 1) / bsizex;
    int blocks_kl = (npairs_kl + bsizey - 1) / bsizey;
    dim3 blocks(blocks_ij, blocks_kl);
    uint16_t *pRt2_kl_ij;
    int8_t *efg_phase;
    cudaGetSymbolAddress((void**)&pRt2_kl_ij, Rt2_kl_ij);
    cudaGetSymbolAddress((void**)&efg_phase, c_Rt2_efg_phase);
    pRt2_kl_ij += offset_for_Rt2_idx(lij, lkl);
    efg_phase += offset_for_Rt2_idx(0, lkl);
    int dm_size = dm_xyz_size * nimgs_uniq_pair;
    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        JKMatrix jmat = {vj+i_dm*dm_size, NULL, dm+i_dm*dm_size, n_dm, 0, omega};
        if (1){//!pbc_md_j_unrolled(envs, &jmat, &bounds, omega)) {
            bounds.qd_ij_max = qd_ij_max + qd_offset_for_threads(npairs_ij, threads_ij);
            bounds.qd_kl_max = qd_kl_max + qd_offset_for_threads(npairs_kl, threads_kl);
            pbc_md_j_kernel<<<blocks, threads, buflen>>>(
                *envs, jmat, bounds, threads_ij, threads_kl, tilex, tiley,
                pRt2_kl_ij, efg_phase);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MD_build_j: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
