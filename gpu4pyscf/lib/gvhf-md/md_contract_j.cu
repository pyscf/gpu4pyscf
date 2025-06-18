/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
#include "gvhf-rys/gamma_inc.cu"
#include "gvhf-md/md_j.cuh"

#define RT2_MAX 9
#define IJ_SIZE 9
#define DM_BLOCK 4

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
inline void iter_Rt_n(double *out, double *Rt, double rx, double ry, double rz, int l,
                      int nsq_per_block, int gout_id, int gout_stride)
{

    int offsets = l*(l+1)*(l+2)*(l+3)/24;
    uint16_t *p1 = c_Rt_idx + offsets - l;
    double *pout = out + nsq_per_block;
    for (int v = gout_id; v < l; v += gout_stride) {
        pout[v*nsq_per_block] = rz * Rt[v*nsq_per_block] + v * Rt[p1[v]*nsq_per_block];
    }
    pout += l * nsq_per_block;
    p1 += l;
    int8_t *tuv_fac = c_Rt_tuv_fac + offsets;

    int n2 = l * (l+1) / 2;
    for (int i = gout_id; i < n2; i += gout_stride) {
        pout[i*nsq_per_block] = ry * Rt[i*nsq_per_block] + tuv_fac[i] * Rt[p1[i]*nsq_per_block];
    }
    pout += n2 * nsq_per_block;
    p1 += n2;
    tuv_fac += n2;

    int n3 = n2 * (l+2) / 3;
    for (int i = gout_id; i < n3; i += gout_stride) {
        pout[i*nsq_per_block] = rx * Rt[i*nsq_per_block] + tuv_fac[i] * Rt[p1[i]*nsq_per_block];
    }
}

__global__
void md_j_1dm_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                     int threadsx, int threadsy, int tilex, int tiley)
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
    int lane_id = t_id % 32;
    int group_id = lane_id / threadsx;
    unsigned int mask = ((1 << threadsx) - 1) << group_id * threadsx;
    int tx = sq_id % threadsx;
    int ty = sq_id / threadsx;
    int threads = nsq_per_block * gout_stride;
    int yslots = gout_stride * threadsy;
    int yslot_id = gout_id * threadsy + ty;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int order = lij + lkl;
    int nf3ijkl = (order+1)*(order+2)*(order+3)/6;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
    int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + nf3kl*bsizey;
    double *Rp_cache = Rq_cache + bsizey*4;
    double *dm_ij_cache = Rp_cache + threadsx*4;
    double *gamma_inc = dm_ij_cache + nf3ij * threadsx;
    double *Rt_buf = gamma_inc + (order+1) * nsq_per_block;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    // zero out all cache;
    for (int n = t_id; n < nf3kl*bsizey + (threadsx+bsizey)*4; n += threads) {
        vj_kl_cache[n] = 0;
    }
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
            Rq_cache[n+3*bsizey] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < tilex; ++batch_ij) {
        int task_ij0 = blockIdx.x * bsizex + batch_ij * threadsx;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = t_id; n < threadsx; n += threads) {
            int task_ij = task_ij0 + n;
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
                Rp_cache[n+0*threadsx] = xij;
                Rp_cache[n+1*threadsx] = yij;
                Rp_cache[n+2*threadsx] = zij;
                Rp_cache[n+3*threadsx] = aij;
            } else {
                Rp_cache[n+3*threadsx] = 1.; // aij
            }
        }
        double fac_sym = PI_FAC;
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
            fac_sym = 0.;
        }
        int pair_ij = pair_ij_mapping[task_ij];
        int ish = pair_ij / nbas;
        int jsh = pair_ij % nbas;
        if (ish == jsh) fac_sym *= .5;
        int ij_loc0 = pair_ij_loc[task_ij];
        for (int n = yslot_id; n < nf3ij; n += yslots) {
            dm_ij_cache[tx+n*threadsx] = dm[ij_loc0+n];
        }
        double vj_ij[IJ_SIZE];
#pragma unroll
        for (int n = 0; n < IJ_SIZE; ++n) {
            vj_ij[n] = 0.;
        }
        for (int batch_kl = 0; batch_kl < tiley; ++batch_kl) {
            int task_kl0 = blockIdx.y * bsizey + batch_kl * threadsy;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+threadsx <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*tilex+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*tiley+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * threadsy;
            int task_kl = task_kl0 + ty;
            double fac = fac_sym;
            if (task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int pair_kl = pair_kl_mapping[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            if (pair_ij_mapping == pair_kl_mapping) {
                if (task_ij == task_kl) fac *= .5;
                if (task_ij < task_kl) fac = 0.;
            }
            __syncthreads();
            double *Rt, *buf;
            if (order % 2 == 0) {
                Rt = Rt_buf + sq_id;
                buf = Rt + nf3ijkl * nsq_per_block;
            } else {
                buf = Rt_buf + sq_id;
                Rt = buf + nf3ijkl * nsq_per_block;
            }
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
            double theta_rr = theta * rr;
            if (gout_id == 0) {
                eval_gamma_inc_fn(gamma_inc, theta_rr, order, sq_id, nsq_per_block);
                double a2 = -2. * theta;
                fac /= aij*akl*sqrt(aij+akl);
                gamma_inc[sq_id] *= fac;
                for (int i = 1; i <= order; i++) {
                    fac *= a2;
                    gamma_inc[sq_id+i*nsq_per_block] *= fac;
                }
                Rt[0] = gamma_inc[sq_id+order*nsq_per_block];
            }
            for (int n = 1; n <= order; ++n) {
                __syncthreads();
                // swap input and output
                double *tmp = buf;
                buf = Rt;
                Rt = tmp;
                if (gout_id == 0) {
                    Rt[0] = gamma_inc[sq_id+(order-n)*nsq_per_block];
                }
                if (n == 1) {
                    if (gout_id == 0) {
                        Rt[1*nsq_per_block] = zpq * buf[0*nsq_per_block];
                        Rt[2*nsq_per_block] = ypq * buf[0*nsq_per_block];
                        Rt[3*nsq_per_block] = xpq * buf[0*nsq_per_block];
                    }
                } else if (n == 2) {
                    if (gout_id == 0) {
                        Rt[1*nsq_per_block] = zpq * buf[0*nsq_per_block];
                        Rt[2*nsq_per_block] = zpq * buf[1*nsq_per_block] + buf[0*nsq_per_block];
                        Rt[3*nsq_per_block] = ypq * buf[0*nsq_per_block];
                        Rt[4*nsq_per_block] = ypq * buf[1*nsq_per_block];
                        Rt[5*nsq_per_block] = ypq * buf[2*nsq_per_block] + buf[0*nsq_per_block];
                        Rt[6*nsq_per_block] = xpq * buf[0*nsq_per_block];
                        Rt[7*nsq_per_block] = xpq * buf[1*nsq_per_block];
                        Rt[8*nsq_per_block] = xpq * buf[2*nsq_per_block];
                        Rt[9*nsq_per_block] = xpq * buf[3*nsq_per_block] + buf[0*nsq_per_block];
                    }
                } else {
                    iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsq_per_block, gout_id, gout_stride);
                }
            }

            Rt = Rt_buf;
            int kl_loc0 = pair_kl_loc[task_kl];
            uint16_t *p1_ij = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lkl];
            uint16_t *p1_kl = Rt2_ij_kl + Rt2_idx_offsets[lij*RT2_MAX+lkl];
            int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lkl];
            for (int k = gout_id; k < nf3kl+gout_id; k += gout_stride) {
                __syncthreads();
                double val = 0.;
                if (k < nf3kl) {
                    double phase = efg_phase[k];
                    int off = k * nf3ij;
                    for (int i = 0; i < nf3ij; ++i) {
                        double s = Rt[sq_id+p1_ij[off+i]*nsq_per_block];
                        val += phase * s * dm_ij_cache[tx+i*threadsx];
                    }
                }
                // reduce ij
                for (int offset = threadsx/2; offset > 0; offset /= 2) {
                    val += __shfl_down_sync(mask, val, offset);
                }
                if (tx == 0 && k < nf3kl) {
                    vj_kl_cache[sq_kl+k*bsizey] += val;
                }
            }

#pragma unroll
            for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                if (i >= nf3ij) break;
                int off = i * nf3kl;
                for (int k = 0; k < nf3kl; ++k) {
                    double s = Rt[sq_id+p1_kl[off+k]*nsq_per_block];
                    vj_ij[n] += efg_phase[k] * s * dm[kl_loc0+k];
                }
            }
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
            if (i >= nf3ij+gout_id) break;
            __syncthreads();
            vj_cache[t_id] = vj_ij[n];
            for (int stride = threadsy/2; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[t_id] += vj_cache[t_id + stride*threadsx];
                }
            }
            __syncthreads();
            if (ty == 0 && i < nf3ij && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+i, vj_cache[t_id]);
            }
        }
    }
    __syncthreads();
    {
        int xslots = threadsx * gout_stride;
        int xslot_id = t_id / threadsy;
        int ty = t_id % threadsy;
        for (int n = xslot_id; n < nf3kl * tiley; n += xslots) {
            int batch_kl = n % tiley;
            int sq_kl = ty + batch_kl * threadsy;
            int task_kl = blockIdx.y * bsizey + sq_kl;
            if (task_kl < npairs_kl) {
                int kl_loc0 = pair_kl_loc[task_kl];
                int kl = n / tiley;
                atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*bsizey]);
            }
        }
    }
}

__global__
void md_j_4dm_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                     int threadsx, int threadsy, int tilex, int tiley,
                     int dm_size)
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
    int lane_id = t_id % 32;
    int group_id = lane_id / threadsx;
    unsigned int mask = ((1 << threadsx) - 1) << group_id * threadsx;
    int tx = sq_id % threadsx;
    int ty = sq_id / threadsx;
    int threads = nsq_per_block * gout_stride;
    int yslots = gout_stride * threadsy;
    int yslot_id = gout_id * threadsy + ty;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int order = lij + lkl;
    int nf3ijkl = (order+1)*(order+2)*(order+3)/6;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
    int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + nf3kl*bsizey * DM_BLOCK;
    double *Rp_cache = Rq_cache + bsizey*4;
    double *dm_ij_cache = Rp_cache + threadsx*4;
    double *gamma_inc = dm_ij_cache + nf3ij * threadsx * DM_BLOCK;
    double *Rt_buf = gamma_inc + (order+1) * nsq_per_block;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    double *dm_ij_cache1 = dm_ij_cache + nf3ij * threadsx;
    double *dm_ij_cache2 = dm_ij_cache + nf3ij * threadsx*2;
    double *dm_ij_cache3 = dm_ij_cache + nf3ij * threadsx*3;
    double *vj_kl_cache1 = vj_kl_cache + nf3kl * bsizey;
    double *vj_kl_cache2 = vj_kl_cache + nf3kl * bsizey*2;
    double *vj_kl_cache3 = vj_kl_cache + nf3kl * bsizey*3;

    // zero out all cache;
    for (int n = t_id; n < nf3kl*bsizey*DM_BLOCK + (threadsx+bsizey)*4; n += threads) {
        vj_kl_cache[n] = 0;
    }
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
            Rq_cache[n+3*bsizey] = 1.;
        }
    }

    double *dm = jk.dm;
    double *vj = jk.vj;
    int remaining_n_dm = jk.n_dm;
    while (remaining_n_dm > 0) {
        double *dm1 = dm + dm_size;
        double *dm2 = dm + dm_size*2;
        double *dm3 = dm + dm_size*3;
        double *vj1 = vj + dm_size;
        double *vj2 = vj + dm_size*2;
        double *vj3 = vj + dm_size*3;
        for (int batch_ij = 0; batch_ij < tilex; ++batch_ij) {
            int task_ij0 = blockIdx.x * bsizex + batch_ij * threadsx;
            if (task_ij0 >= npairs_ij) {
                continue;
            }
            __syncthreads();
            for (int n = t_id; n < threadsx; n += threads) {
                int task_ij = task_ij0 + n;
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
                    Rp_cache[n+0*threadsx] = xij;
                    Rp_cache[n+1*threadsx] = yij;
                    Rp_cache[n+2*threadsx] = zij;
                    Rp_cache[n+3*threadsx] = aij;
                } else {
                    Rp_cache[n+3*threadsx] = 1.;
                }
            }
            double fac_sym = PI_FAC;
            int task_ij = task_ij0 + tx;
            if (task_ij >= npairs_ij) {
                task_ij = task_ij0;
                fac_sym = 0.;
            }
            int pair_ij = pair_ij_mapping[task_ij];
            int ish = pair_ij / nbas;
            int jsh = pair_ij % nbas;
            if (ish == jsh) fac_sym *= .5;

            int ij_loc0 = pair_ij_loc[task_ij];
            int nf3ij_dm = nf3ij * min(remaining_n_dm, DM_BLOCK);
            for (int n = yslot_id; n < nf3ij_dm; n += yslots) {
                int i = n / nf3ij;
                dm_ij_cache[tx+n*threadsx] = dm[i*dm_size+ij_loc0+n];
            }
            double vj_ij[IJ_SIZE*DM_BLOCK];
#pragma unroll
            for (int n = 0; n < IJ_SIZE*DM_BLOCK; ++n) {
                vj_ij[n] = 0.;
            }
            for (int batch_kl = 0; batch_kl < tiley; ++batch_kl) {
                int task_kl0 = blockIdx.y * bsizey + batch_kl * threadsy;
                if (task_kl0 >= npairs_kl) {
                    continue;
                }
                if (pair_ij_mapping == pair_kl_mapping && task_ij0+threadsx <= task_kl0) {
                    continue;
                }
                int pair_ij0 = pair_ij_mapping[task_ij0];
                int pair_kl0 = pair_kl_mapping[task_kl0];
                if (qd_ij_max[blockIdx.x*tilex+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                    qd_kl_max[blockIdx.y*tiley+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                    continue;
                }

                int sq_kl = ty + batch_kl * threadsy;
                int task_kl = task_kl0 + ty;
                double fac = fac_sym;
                if (task_kl >= npairs_kl) {
                    task_kl = task_kl0;
                    fac = 0.;
                }
                int pair_kl = pair_kl_mapping[task_kl];
                int ksh = pair_kl / nbas;
                int lsh = pair_kl % nbas;
                if (ksh == lsh) fac *= .5;
                if (pair_ij_mapping == pair_kl_mapping) {
                    if (task_ij == task_kl) fac *= .5;
                    if (task_ij < task_kl) fac = 0.;
                }
                __syncthreads();
                double *Rt, *buf;
                if (order % 2 == 0) {
                    Rt = Rt_buf + sq_id;
                    buf = Rt + nf3ijkl * nsq_per_block;
                } else {
                    buf = Rt_buf + sq_id;
                    Rt = buf + nf3ijkl * nsq_per_block;
                }
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
                double theta_rr = theta * rr;
                if (gout_id == 0) {
                    eval_gamma_inc_fn(gamma_inc, theta_rr, order, sq_id, nsq_per_block);
                    double a2 = -2. * theta;
                    fac /= aij*akl*sqrt(aij+akl);
                    gamma_inc[sq_id] *= fac;
                    for (int i = 1; i <= order; i++) {
                        fac *= a2;
                        gamma_inc[sq_id+i*nsq_per_block] *= fac;
                    }
                    Rt[0] = gamma_inc[sq_id+order*nsq_per_block];
                }
                for (int n = 1; n <= order; ++n) {
                    __syncthreads();
                    // swap input and output
                    double *tmp = buf;
                    buf = Rt;
                    Rt = tmp;
                    if (gout_id == 0) {
                        Rt[0] = gamma_inc[sq_id+(order-n)*nsq_per_block];
                    }
                    if (n == 1) {
                        if (gout_id == 0) {
                            Rt[1*nsq_per_block] = zpq * buf[0*nsq_per_block];
                            Rt[2*nsq_per_block] = ypq * buf[0*nsq_per_block];
                            Rt[3*nsq_per_block] = xpq * buf[0*nsq_per_block];
                        }
                    } else if (n == 2) {
                        if (gout_id == 0) {
                            Rt[1*nsq_per_block] = zpq * buf[0*nsq_per_block];
                            Rt[2*nsq_per_block] = zpq * buf[1*nsq_per_block] + buf[0*nsq_per_block];
                            Rt[3*nsq_per_block] = ypq * buf[0*nsq_per_block];
                            Rt[4*nsq_per_block] = ypq * buf[1*nsq_per_block];
                            Rt[5*nsq_per_block] = ypq * buf[2*nsq_per_block] + buf[0*nsq_per_block];
                            Rt[6*nsq_per_block] = xpq * buf[0*nsq_per_block];
                            Rt[7*nsq_per_block] = xpq * buf[1*nsq_per_block];
                            Rt[8*nsq_per_block] = xpq * buf[2*nsq_per_block];
                            Rt[9*nsq_per_block] = xpq * buf[3*nsq_per_block] + buf[0*nsq_per_block];
                        }
                    } else {
                        iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsq_per_block, gout_id, gout_stride);
                    }
                }

                Rt = Rt_buf;
                int sq_kl1 = sq_kl + nf3kl * bsizey;
                int sq_kl2 = sq_kl + nf3kl * bsizey*2;
                int sq_kl3 = sq_kl + nf3kl * bsizey*3;
                int kl_loc0 = pair_kl_loc[task_kl];
                uint16_t *p1_ij = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lkl];
                uint16_t *p1_kl = Rt2_ij_kl + Rt2_idx_offsets[lij*RT2_MAX+lkl];
                int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lkl];
                switch (remaining_n_dm) {
                case 1:
                    for (int k = gout_id; k < nf3kl+gout_id; k += gout_stride) {
                        __syncthreads();
                        double val = 0.;
                        if (k < nf3kl) {
                            double phase = efg_phase[k];
                            int off = k * nf3ij;
                            for (int i = 0; i < nf3ij; ++i) {
                                double s = Rt[sq_id+p1_ij[off+i]*nsq_per_block];
                                val += phase * s * dm_ij_cache[tx+i*threadsx];
                            }
                        }
                        for (int offset = threadsx/2; offset > 0; offset /= 2) {
                            val += __shfl_down_sync(mask, val, offset);
                        }
                        if (tx == 0 && k < nf3kl) {
                            vj_kl_cache[sq_kl+k*bsizey] += val;
                        }
                    }
#pragma unroll
                    for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                        if (i >= nf3ij) break;
                        int off = i * nf3kl;
                        for (int k = 0; k < nf3kl; ++k) {
                            double s = Rt[sq_id+p1_kl[off+k]*nsq_per_block];
                            vj_ij[n] += efg_phase[k] * s * dm[kl_loc0+k];
                        }
                    }
                    break;
                case 2:
                    for (int k = gout_id; k < nf3kl+gout_id; k += gout_stride) {
                        __syncthreads();
                        double val0 = 0.;
                        double val1 = 0.;
                        if (k < nf3kl) {
                            double phase = efg_phase[k];
                            int off = k * nf3ij;
                            for (int i = 0; i < nf3ij; ++i) {
                                double s = Rt[sq_id+p1_ij[off+i]*nsq_per_block];
                                double phase_s = phase * s;
                                val0 += phase_s * dm_ij_cache [tx+i*threadsx];
                                val1 += phase_s * dm_ij_cache1[tx+i*threadsx];
                            }
                        }
                        for (int offset = threadsx/2; offset > 0; offset /= 2) {
                            val0 += __shfl_down_sync(mask, val0, offset);
                            val1 += __shfl_down_sync(mask, val1, offset);
                        }
                        if (tx == 0 && k < nf3kl) {
                            vj_kl_cache[sq_kl +k*bsizey] += val0;
                            vj_kl_cache[sq_kl1+k*bsizey] += val1;
                        }
                    }
#pragma unroll
                    for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                        if (i >= nf3ij) break;
                        int off = i * nf3kl;
                        for (int k = 0; k < nf3kl; ++k) {
                            double s = Rt[sq_id+p1_kl[off+k]*nsq_per_block];
                            double phase_s = efg_phase[k] * s;
                            vj_ij[n        ] += phase_s * dm [kl_loc0+k];
                            vj_ij[n+IJ_SIZE] += phase_s * dm1[kl_loc0+k];
                        }
                    }
                    break;
                case 3:
                    dm3 = dm2;
                default:
                    for (int k = gout_id; k < nf3kl+gout_id; k += gout_stride) {
                        __syncthreads();
                        double val0 = 0.;
                        double val1 = 0.;
                        double val2 = 0.;
                        double val3 = 0.;
                        if (k < nf3kl) {
                            double phase = efg_phase[k];
                            int off = k * nf3ij;
                            for (int i = 0; i < nf3ij; ++i) {
                                double s = Rt[sq_id+p1_ij[off+i]*nsq_per_block];
                                double phase_s = phase * s;
                                val0 += phase_s * dm_ij_cache [tx+i*threadsx];
                                val1 += phase_s * dm_ij_cache1[tx+i*threadsx];
                                val2 += phase_s * dm_ij_cache2[tx+i*threadsx];
                                val3 += phase_s * dm_ij_cache3[tx+i*threadsx];
                            }
                        }
                        // reduce along ij
                        for (int offset = threadsx/2; offset > 0; offset /= 2) {
                            val0 += __shfl_down_sync(mask, val0, offset);
                            val1 += __shfl_down_sync(mask, val1, offset);
                            val2 += __shfl_down_sync(mask, val2, offset);
                            val3 += __shfl_down_sync(mask, val3, offset);
                        }
                        if (tx == 0 && k < nf3kl) {
                            vj_kl_cache[sq_kl +k*bsizey] += val0;
                            vj_kl_cache[sq_kl1+k*bsizey] += val1;
                            vj_kl_cache[sq_kl2+k*bsizey] += val2;
                            vj_kl_cache[sq_kl3+k*bsizey] += val3;
                        }
                    }

#pragma unroll
                    for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                        if (i >= nf3ij) break;
                        int off = i * nf3kl;
                        for (int k = 0; k < nf3kl; ++k) {
                            double s = Rt[sq_id+p1_kl[off+k]*nsq_per_block];
                            double phase_s = efg_phase[k] * s;
                            vj_ij[n+IJ_SIZE*0] += phase_s * dm [kl_loc0+k];
                            vj_ij[n+IJ_SIZE*1] += phase_s * dm1[kl_loc0+k];
                            vj_ij[n+IJ_SIZE*2] += phase_s * dm2[kl_loc0+k];
                            vj_ij[n+IJ_SIZE*3] += phase_s * dm3[kl_loc0+k];
                        }
                    }
                }
            }

            double *vj_cache = Rp_cache;
            double *vj_cache1 = vj_cache + threads;
            double *vj_cache2 = vj_cache + threads*2;
            double *vj_cache3 = vj_cache + threads*3;
            switch (remaining_n_dm) {
            case 1:
#pragma unroll
                for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                    if (i >= nf3ij+gout_id) break;
                    __syncthreads();
                    vj_cache[t_id] = vj_ij[n];
                    for (int stride = threadsy/2; stride > 0; stride /= 2) {
                        __syncthreads();
                        if (ty < stride) {
                            vj_cache[t_id] += vj_cache[t_id + stride*threadsx];
                        }
                    }
                    __syncthreads();
                    if (ty == 0 && i < nf3ij && task_ij0+tx < npairs_ij) {
                        atomicAdd(vj+ij_loc0+i, vj_cache[t_id]);
                    }
                }
                break;
            case 2:
#pragma unroll
                for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                    if (i >= nf3ij+gout_id) break;
                    __syncthreads();
                    vj_cache [t_id] = vj_ij[n];
                    vj_cache1[t_id] = vj_ij[n+IJ_SIZE];
                    for (int stride = threadsy/2; stride > 0; stride /= 2) {
                        __syncthreads();
                        if (ty < stride) {
                            vj_cache [t_id] += vj_cache [t_id + stride*threadsx];
                            vj_cache1[t_id] += vj_cache1[t_id + stride*threadsx];
                        }
                    }
                    __syncthreads();
                    if (ty == 0 && i < nf3ij && task_ij0+tx < npairs_ij) {
                        atomicAdd(vj +ij_loc0+i, vj_cache [t_id]);
                        atomicAdd(vj1+ij_loc0+i, vj_cache1[t_id]);
                    }
                }
                break;
            default:
#pragma unroll
                for (int n = 0, i = gout_id; n < IJ_SIZE; ++n, i += gout_stride) {
                    if (i >= nf3ij+gout_id) break;
                    __syncthreads();
                    vj_cache [t_id] = vj_ij[n];
                    vj_cache1[t_id] = vj_ij[n+IJ_SIZE];
                    vj_cache2[t_id] = vj_ij[n+IJ_SIZE*2];
                    vj_cache3[t_id] = vj_ij[n+IJ_SIZE*3];
                    for (int stride = threadsy/2; stride > 0; stride /= 2) {
                        __syncthreads();
                        if (ty < stride) {
                            vj_cache [t_id] += vj_cache [t_id + stride*threadsx];
                            vj_cache1[t_id] += vj_cache1[t_id + stride*threadsx];
                            vj_cache2[t_id] += vj_cache2[t_id + stride*threadsx];
                            vj_cache3[t_id] += vj_cache3[t_id + stride*threadsx];
                        }
                    }
                    __syncthreads();
                    if (ty == 0 && i < nf3ij && task_ij0+tx < npairs_ij) {
                        atomicAdd(vj +ij_loc0+i, vj_cache [t_id]);
                        atomicAdd(vj1+ij_loc0+i, vj_cache1[t_id]);
                        atomicAdd(vj2+ij_loc0+i, vj_cache2[t_id]);
                        if (remaining_n_dm > 3) {
                            atomicAdd(vj3+ij_loc0+i, vj_cache3[t_id]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        {
            int xslots = threadsx * gout_stride;
            int xslot_id = t_id / threadsy;
            int ty = t_id % threadsy;
            for (int n = xslot_id; n < nf3kl * tiley; n += xslots) {
                int batch_kl = n % tiley;
                int sq_kl = ty + batch_kl * threadsy;
                int task_kl = blockIdx.y * bsizey + sq_kl;
                if (task_kl < npairs_kl) {
                    int kl_loc0 = pair_kl_loc[task_kl];
                    int kl = n / tiley;
                    switch (remaining_n_dm) {
                    case 1:
                        atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*bsizey]);
                        break;
                    case 2:
                        atomicAdd(vj +kl_loc0+kl, vj_kl_cache [sq_kl+kl*bsizey]);
                        atomicAdd(vj1+kl_loc0+kl, vj_kl_cache1[sq_kl+kl*bsizey]);
                        break;
                    case 3:
                        atomicAdd(vj +kl_loc0+kl, vj_kl_cache [sq_kl+kl*bsizey]);
                        atomicAdd(vj1+kl_loc0+kl, vj_kl_cache1[sq_kl+kl*bsizey]);
                        atomicAdd(vj2+kl_loc0+kl, vj_kl_cache2[sq_kl+kl*bsizey]);
                        break;
                    default:
                        atomicAdd(vj +kl_loc0+kl, vj_kl_cache [sq_kl+kl*bsizey]);
                        atomicAdd(vj1+kl_loc0+kl, vj_kl_cache1[sq_kl+kl*bsizey]);
                        atomicAdd(vj2+kl_loc0+kl, vj_kl_cache2[sq_kl+kl*bsizey]);
                        atomicAdd(vj3+kl_loc0+kl, vj_kl_cache3[sq_kl+kl*bsizey]);
                    }
                }
            }
        }
        switch (remaining_n_dm) {
        case 1:
            remaining_n_dm -= 1;
            break;
        case 2:
            remaining_n_dm -= 2;
            break;
        default:
            remaining_n_dm -= 4;
        }
        dm += dm_size * 4;
        vj += dm_size * 4;
    }
}
