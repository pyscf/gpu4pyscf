/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
#include <cuda_runtime.h>

#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/gamma_inc.cu"

#define RT2_MAX 9

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
    __syncthreads();
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

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void md_j_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                 int threadsx, int threadsy, int tilex, int tiley)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int bsizex = threadsx * tilex;
    int bsizey = threadsy * tiley;
    int nsq_per_block = threadsx * threadsy;
    int gout_stride = blockDim.x / nsq_per_block;
    int task_ij0 = blockIdx.x * bsizex;
    int task_kl0 = blockIdx.y * bsizey;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int t_id = threadIdx.x;
    int sq_id = t_id % nsq_per_block;
    int gout_id = t_id / nsq_per_block;
    int tx = sq_id % threadsx;
    int ty = sq_id / threadsx;
    int threads = nsq_per_block * gout_stride;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int order = lij + lkl;
    int nf3ijkl = (order+1)*(order+2)*(order+3)/6;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
    int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + (order+1) * nsq_per_block;
    double *Rq_cache = Rp_cache + bsizex*4;
    double *vj_ij_cache = Rq_cache + bsizey*4;
    double *vj_kl_cache = vj_ij_cache + nf3ij * bsizex;

    // zero out all cache;
    for (int n = t_id; n < (bsizex*4 + bsizey*4 + nf3ij*bsizex + nf3kl*bsizey); n += threads) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();
    for (int n = t_id; n < bsizex; n += threads) {
        int task_ij = blockIdx.x * bsizex + n;
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
            Rp_cache[n+0*bsizex] = xij;
            Rp_cache[n+1*bsizex] = yij;
            Rp_cache[n+2*bsizex] = zij;
            Rp_cache[n+3*bsizex] = aij;
        } else {
            Rp_cache[n+3*bsizex] = 1.;
        }
    }
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
    //for (int n = ty+threadsy*gout_id; n < nf3ij*tilex; n += threadsy*gout_stride) {
    //    int i = n / tilex;
    //    int tile = n % tilex;
    //    int task_ij = blockIdx.x * bsizex + tile * threadsx + tx;
    //    if (task_ij < npairs_ij) {
    //        int pair_ij = pair_ij_mapping[task_ij];
    //        int dm_ij_pair0 = dm_pair_loc[pair_ij];
    //        int sq_ij = tx + tile * threadsx;
    //        dm_ij_cache[sq_ij+i*bsizex] = dm[dm_ij_pair0+i];
    //    }
    //}
    //for (int n = tx+threadsx*gout_id; n < nf3kl*tiley; n += threadsx*gout_stride) {
    //    int i = n / tiley;
    //    int tile = n % tiley;
    //    int task_kl = blockIdx.y * bsizey + tile * threadsy + ty;
    //    if (task_kl < npairs_kl) {
    //        int pair_kl = pair_kl_mapping[task_kl];
    //        int dm_kl_pair0 = dm_pair_loc[pair_kl];
    //        int sq_kl = ty + tile * threadsy;
    //        dm_kl_cache[sq_kl+i*bsizey] = dm[dm_kl_pair0+i];
    //    }
    //}

    for (int batch_ij = 0; batch_ij < tilex; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < tiley; ++batch_kl) {
        int task_ij0 = blockIdx.x * bsizex + batch_ij * threadsx;
        int task_kl0 = blockIdx.y * bsizey + batch_kl * threadsy;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * threadsx;
        int sq_kl = ty + batch_kl * threadsy;
        int task_ij = task_ij0 + tx;
        int task_kl = task_kl0 + ty;
        double fac_sym = PI_FAC;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
            fac_sym = 0.;
        }
        if (task_kl >= npairs_kl) {
            task_kl = task_kl0;
            fac_sym = 0.;
        }
        int pair_ij = pair_ij_mapping[task_ij];
        int pair_kl = pair_kl_mapping[task_kl];

        int ish = pair_ij / nbas;
        int jsh = pair_ij % nbas;
        int ksh = pair_kl / nbas;
        int lsh = pair_kl % nbas;
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (pair_ij_mapping == pair_kl_mapping) {
            if (task_ij == task_kl) fac_sym *= .5;
            // TODO: skip certain blocks when task_ij < task_kl
            if (task_ij < task_kl) fac_sym = 0.;
        }
        int dm_ij_pair0 = dm_pair_loc[pair_ij];
        int dm_kl_pair0 = dm_pair_loc[pair_kl];
        double *Rt, *buf;
        if (order % 2 == 0) {
            Rt = vj_kl_cache + nf3kl*bsizey + sq_id;
            buf = Rt + nf3ijkl * nsq_per_block;
        } else {
            buf = vj_kl_cache + nf3kl*bsizey + sq_id;
            Rt = buf + nf3ijkl * nsq_per_block;
        }
        __syncthreads();
        double xij = Rp_cache[sq_ij+0*bsizex];
        double yij = Rp_cache[sq_ij+1*bsizex];
        double zij = Rp_cache[sq_ij+2*bsizex];
        double aij = Rp_cache[sq_ij+3*bsizex];
        double xkl = Rq_cache[sq_kl+0*bsizey];
        double ykl = Rq_cache[sq_kl+1*bsizey];
        double zkl = Rq_cache[sq_kl+2*bsizey];
        double akl = Rq_cache[sq_kl+3*bsizey];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
        double xpq = xij - xkl;
        double ypq = yij - ykl;
        double zpq = zij - zkl;
        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
        double theta = aij * akl / (aij + akl);
        double theta_rr = theta * rr;
        if (gout_id == 0) {
            eval_gamma_inc_fn(gamma_inc, theta_rr, order, sq_id, nsq_per_block);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= order; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*nsq_per_block] *= fac;
            }
            Rt[0] = gamma_inc[sq_id+order*nsq_per_block];
        }
        for (int n = 1; n <= order; ++n) {
            // swap input and output
            double *tmp = buf;
            buf = Rt;
            Rt = tmp;
            if (gout_id == 0) {
                Rt[0] = gamma_inc[sq_id+(order-n)*nsq_per_block];
            }
            switch (n) {
            case 1:
                if (gout_id == 0) {
                    Rt[1*nsq_per_block] = zpq * buf[0*nsq_per_block];
                    Rt[2*nsq_per_block] = ypq * buf[0*nsq_per_block];
                    Rt[3*nsq_per_block] = xpq * buf[0*nsq_per_block];
                }
                break;
            case 2:
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
                break;
            case 3:
                if (gout_id == 0) {
                    Rt[1*nsq_per_block] = zpq * buf[0*nsq_per_block];
                    Rt[2*nsq_per_block] = zpq * buf[1*nsq_per_block] + buf[0*nsq_per_block];
                    Rt[3*nsq_per_block] = zpq * buf[2*nsq_per_block] + 2 * buf[1*nsq_per_block];
                    Rt[4*nsq_per_block] = ypq * buf[0*nsq_per_block];
                    Rt[5*nsq_per_block] = ypq * buf[1*nsq_per_block];
                    Rt[6*nsq_per_block] = ypq * buf[2*nsq_per_block];
                    Rt[7*nsq_per_block] = ypq * buf[3*nsq_per_block] + buf[0*nsq_per_block];
                    Rt[8*nsq_per_block] = ypq * buf[4*nsq_per_block] + buf[1*nsq_per_block];
                    Rt[9*nsq_per_block] = ypq * buf[5*nsq_per_block] + 2 * buf[3*nsq_per_block];
                    Rt[10*nsq_per_block] = xpq * buf[0*nsq_per_block];
                    Rt[11*nsq_per_block] = xpq * buf[1*nsq_per_block];
                    Rt[12*nsq_per_block] = xpq * buf[2*nsq_per_block];
                    Rt[13*nsq_per_block] = xpq * buf[3*nsq_per_block];
                    Rt[14*nsq_per_block] = xpq * buf[4*nsq_per_block];
                    Rt[15*nsq_per_block] = xpq * buf[5*nsq_per_block];
                    Rt[16*nsq_per_block] = xpq * buf[6*nsq_per_block] + buf[0*nsq_per_block];
                    Rt[17*nsq_per_block] = xpq * buf[7*nsq_per_block] + buf[1*nsq_per_block];
                    Rt[18*nsq_per_block] = xpq * buf[8*nsq_per_block] + buf[3*nsq_per_block];
                    Rt[19*nsq_per_block] = xpq * buf[9*nsq_per_block] + 2 * buf[6*nsq_per_block];
                }
                break;
            default: iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsq_per_block, gout_id, gout_stride);
            }
        }

        Rt = vj_kl_cache + nf3kl*bsizey;
        double *vj_cache = Rt + nf3ijkl * nsq_per_block;
        uint16_t *p1 = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lkl];
        int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lkl];
        for (int k = gout_id; k < nf3kl+gout_id; k += gout_stride) {
            __syncthreads();
            double vj_kl = 0.;
            if (k < nf3kl) {
                double fac = efg_phase[k];
                int off = k * nf3ij;
                for (int i = 0; i < nf3ij; ++i) {
                    double s = Rt[sq_id+p1[off+i]*nsq_per_block];
                    vj_kl += fac * s * dm[dm_ij_pair0+i];
                }
            }
            vj_cache[t_id] = vj_kl;
            for (int stride = threadsx/2; stride > 0; stride /= 2) {
                __syncthreads();
                if (tx < stride) {
                    vj_cache[t_id] += vj_cache[t_id + stride];
                }
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl && k < nf3kl) {
                vj_kl_cache[sq_kl+k*bsizey] += vj_cache[t_id];
            }
        }

        p1 = Rt2_ij_kl + Rt2_idx_offsets[lij*RT2_MAX+lkl];
        for (int i = gout_id; i < nf3ij+gout_id; i += gout_stride) {
            __syncthreads();
            double vj_ij = 0.;
            if (i < nf3ij) {
                int off = i * nf3kl;
                for (int k = 0; k < nf3kl; ++k) {
                    double s = Rt[sq_id+p1[off+k]*nsq_per_block];
                    vj_ij += efg_phase[k] * s * dm[dm_kl_pair0+k];
                }
            }
            vj_cache[t_id] = vj_ij;
            for (int stride = threadsy/2; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[t_id] += vj_cache[t_id + stride*threadsx];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij && i < nf3ij) {
                vj_ij_cache[sq_ij+i*bsizex] += vj_cache[t_id];
            }
        }
        __syncthreads();
    } }

    for (int n = ty+threadsy*gout_id; n < nf3ij*tilex; n += threadsy*gout_stride) {
        int i = n / tilex;
        int tile = n % tilex;
        int task_ij = blockIdx.x * bsizex + tile * threadsx + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * threadsx;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*bsizex]);
        }
    }
    for (int n = tx+threadsx*gout_id; n < nf3kl*tiley; n += threadsx*gout_stride) {
        int i = n / tiley;
        int tile = n % tiley;
        int task_kl = blockIdx.y * bsizey + tile * threadsy + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * threadsy;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*bsizey]);
        }
    }
}
