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
#define KL_SIZE 28

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
void md_j_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
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
        task_ij0 < task_kl0) {
        return;
    }

    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int nsq_per_block = blockDim.x;
    int t_id = gout_id * nsq_per_block + sq_id;
    int lane_id = t_id % 32;
    int group_id = lane_id / threadsx;
    unsigned int mask = ((1 << threadsx) - 1) << group_id * threadsx;
    int tx = sq_id % threadsx;
    int ty = sq_id / threadsx;
    int threads = nsq_per_block * gout_stride;
    int xslots = gout_stride * threadsx;
    int yslots = gout_stride * threadsy;
    int xslot_id = gout_id * threadsx + tx;
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
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + (order+1) * nsq_per_block;
    double *Rq_cache = Rp_cache + threadsx*4;
    double *dm_ij_cache = Rq_cache + bsizey*4;
    double *dm_kl_cache = dm_ij_cache + nf3ij * threadsx;
    double *vj_ij_cache = dm_kl_cache + nf3kl * threadsy;
    double *vj_kl_cache = vj_ij_cache + nf3ij * threadsx;
    double *Rt_buf = vj_kl_cache + nf3kl*threadsy;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    // zero out all cache;
    for (int n = t_id; n < (threadsx+bsizey)*4; n += threads) {
        Rp_cache[n] = 0.;
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

#if 1
    //register double vj_kl[KL_SIZE];
    //for (int n = 0; n < KL_SIZE; ++n) {
    //    vj_kl[n] = 0;
    //}

    int kl_counts = nf3kl * tiley;
    register double dm_kl[KL_SIZE];
    for (int n = 0; n < KL_SIZE; ++n) {
        dm_kl[n] = 0;
    }
    for (int k = 0; k <= KL_SIZE; ++k) {
        int n = k * xslots + xslot_id;
        if (n >= kl_counts) break;
        int tile = n / nf3kl;
        int task_kl = blockIdx.y * bsizey + tile * threadsy + ty;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n % nf3kl;
            dm_kl[k] = dm[kl_loc0+kl];
        }
    }
#endif
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
        for (int n = yslot_id; n < nf3ij; n += yslots) {
            dm_ij_cache[tx+n*threadsx] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*threadsx] = 0;
        }
        for (int batch_kl = 0; batch_kl < tiley; ++batch_kl) {
            int task_kl0 = blockIdx.y * bsizey + batch_kl * threadsy;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping) {
                if (task_ij0 < task_kl0) continue;
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
#if 1
            // load dm_kl_cache from dm_kl regisers of each thread
            int addr0 = batch_kl * nf3kl;
            int addr1 = addr0 + nf3kl;
            for (int n = xslot_id; n < nf3kl; n += xslots) {
                vj_kl_cache[ty+n*threadsy] = 0;
            }
            switch (addr0 / xslots) {
            case 0:
                for (int n = 0; n <  3; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 1:
                for (int n = 1; n <= 4; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 2:
                for (int n = 2; n <= 5; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 3:
                for (int n = 3; n <= 6; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 4:
                for (int n = 4; n <= 7; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 5:
                for (int n = 5; n <= 8; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 6:
                for (int n = 6; n <= 9; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 7:
                for (int n = 7; n <= 10; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 8:
                for (int n = 8; n <= 11; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 9:
                for (int n = 9; n <= 12; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 10:
                for (int n = 10; n <= 13; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 11:
                for (int n = 11; n <= 14; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 12:
                for (int n = 12; n <= 15; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 13:
                for (int n = 13; n <= 16; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 14:
                for (int n = 14; n <= 17; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 15:
                for (int n = 15; n <= 18; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 16:
                for (int n = 16; n <= 19; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 17:
                for (int n = 17; n <= 20; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 18:
                for (int n = 18; n <= 21; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 19:
                for (int n = 19; n <= 22; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 20:
                for (int n = 20; n <= 23; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 21:
                for (int n = 21; n <= 24; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 22:
                for (int n = 22; n <= 25; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 23:
                for (int n = 23; n <= 26; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 24:
                for (int n = 24; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 25:
                for (int n = 25; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 26:
                for (int n = 26; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 27:
                {
                    int n = 27;
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            default:
                for (int n = xslot_id; n < nf3kl; n += xslots) {
                    // Assign a special value to dm cache. When the output shows
                    // values ~1e200, this indicates that tiley size is too big.
                    // j_engine scheme function should be adjusted.
                    dm_kl_cache[ty+n*threadsy] = -1e200;
                }
            }
#else
            {
                int xslots = gout_stride * threadsx;
                int xslot_id = gout_id * threadsx + tx;
                int kl_loc0 = pair_kl_loc[task_kl];
                for (int n = xslot_id; n < nf3kl; n += xslots) {
                    dm_kl_cache[ty+n*threadsy] = dm[kl_loc0+n];
                    vj_kl_cache[ty+n*threadsy] = 0;
                }
            }
#endif
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
                default:
                    iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsq_per_block, gout_id, gout_stride);
                }
            }

            Rt = Rt_buf;
            double *vj_cache = Rt + nf3ijkl * nsq_per_block;
            uint16_t *p1 = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lkl];
            int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lkl];
            for (int k = gout_id; k < nf3kl+gout_id; k += gout_stride) {
                __syncthreads();
                double val = 0.;
                if (k < nf3kl) {
                    double phase = efg_phase[k];
                    int off = k * nf3ij;
                    for (int i = 0; i < nf3ij; ++i) {
                        double s = Rt[sq_id+p1[off+i]*nsq_per_block];
                        val += phase * s * dm_ij_cache[tx+i*threadsx];
                    }
                }
                //vj_cache[t_id] = val;
                //for (int stride = threadsx/2; stride > 0; stride /= 2) {
                //    __syncthreads();
                //    if (tx < stride) {
                //        vj_cache[t_id] += vj_cache[t_id + stride];
                //    }
                //}
                //__syncthreads();
                //if (tx == 0 && k < nf3kl) {
                //    vj_kl_cache[ty+k*threadsy] += vj_cache[t_id];
                //}
                for (int offset = threadsx/2; offset > 0; offset /= 2) {
                    val += __shfl_down_sync(mask, val, offset);
                }
                if (tx == 0 && k < nf3kl) {
                    vj_kl_cache[ty+k*threadsy] += val;
                }
            }

            p1 = Rt2_ij_kl + Rt2_idx_offsets[lij*RT2_MAX+lkl];
            for (int i = gout_id; i < nf3ij+gout_id; i += gout_stride) {
                __syncthreads();
                double val = 0.;
                if (i < nf3ij) {
                    int off = i * nf3kl;
                    for (int k = 0; k < nf3kl; ++k) {
                        double s = Rt[sq_id+p1[off+k]*nsq_per_block];
                        val += efg_phase[k] * s * dm_kl_cache[ty+k*threadsy];
                    }
                }
                vj_cache[t_id] = val;
                for (int stride = threadsy/2; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache[t_id] += vj_cache[t_id + stride*threadsx];
                    }
                }
                __syncthreads();
                if (ty == 0 && i < nf3ij) {
                    vj_ij_cache[tx+i*threadsx] += vj_cache[t_id];
                }
            }
            __syncthreads();
#if 0
            switch (addr0 / xslots) {
            case 0:
                for (int n = 0; n <  3; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 1:
                for (int n = 1; n <= 4; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 2:
                for (int n = 2; n <= 5; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 3:
                for (int n = 3; n <= 6; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 4:
                for (int n = 4; n <= 7; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 5:
                for (int n = 5; n <= 8; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 6:
                for (int n = 6; n <= 9; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 7:
                for (int n = 7; n <= 10; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 8:
                for (int n = 8; n <= 11; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 9:
                for (int n = 9; n <= 12; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 10:
                for (int n = 10; n <= 13; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 11:
                for (int n = 11; n <= 14; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 12:
                for (int n = 12; n <= 15; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 13:
                for (int n = 13; n <= 16; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 14:
                for (int n = 14; n <= 17; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 15:
                for (int n = 15; n <= 18; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 16:
                for (int n = 16; n <= 19; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 17:
                for (int n = 17; n <= 20; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 18:
                for (int n = 18; n <= 21; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 19:
                for (int n = 19; n <= 22; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 20:
                for (int n = 20; n <= 23; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 21:
                for (int n = 21; n <= 24; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 22:
                for (int n = 22; n <= 25; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 23:
                for (int n = 23; n <= 26; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 24:
                for (int n = 24; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 25:
                for (int n = 25; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 26:
                for (int n = 26; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            case 27:
                {
                    int n = 27;
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        vj_kl[n] += vj_kl_cache[ty+(addr-addr0)*threadsy];
                    }
                }
                break;
            default:
                vj_kl[27] = 1e200;
            }
#else
            if (task_kl0+ty < npairs_kl) {
                int kl_loc0 = pair_kl_loc[task_kl];
                for (int n = xslot_id; n < nf3kl; n += xslots) {
                    atomicAdd(vj+kl_loc0+n, vj_kl_cache[ty+n*threadsy]);
                }
            }
#endif
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_ij_loc[task_ij];
            for (int n = yslot_id; n < nf3ij; n += yslots) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*threadsx]);
            }
        }
    }
#if 0
    {
        int kl_counts = nf3kl * tiley;
        for (int k = 0; k <= KL_SIZE; ++k) {
            int n = k * xslots + xslot_id;
            if (n >= kl_counts) break;
            int tile = n / nf3kl;
            int task_kl = blockIdx.y * bsizey + tile * threadsy + ty;
            if (task_kl < npairs_kl) {
                int kl_loc0 = pair_kl_loc[task_kl];
                int kl = n % nf3kl;
                atomicAdd(vj+kl_loc0+kl, vj_kl[k]);
            }
        }
    }
#endif
}

// 4-fold permutation symmetry
__global__
void md_j_s4_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
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

    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int nsq_per_block = blockDim.x;
    int t_id = gout_id * nsq_per_block + sq_id;
    int tx = sq_id % threadsx;
    int ty = sq_id / threadsx;
    int threads = nsq_per_block * gout_stride;
    int xslots = gout_stride * threadsx;
    int yslots = gout_stride * threadsy;
    int xslot_id = gout_id * threadsx + tx;
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
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + (order+1) * nsq_per_block;
    double *Rq_cache = Rp_cache + threadsx*4;
    double *dm_kl_cache = Rq_cache + bsizey*4;
    double *vj_ij_cache = dm_kl_cache + nf3kl * threadsy;
    double *Rt_buf = vj_ij_cache + nf3ij * threadsx;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    // zero out all cache;
    for (int n = t_id; n < (threadsx+bsizey)*4; n += threads) {
        Rp_cache[n] = 0.;
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

    register double dm_kl[KL_SIZE];
    for (int n = 0; n < KL_SIZE; ++n) {
        dm_kl[n] = 0;
    }

    int kl_counts = nf3kl * tiley;
    for (int k = 0; k <= KL_SIZE; ++k) {
        int n = k * xslots + xslot_id;
        if (n >= kl_counts) break;
        int tile = n / nf3kl;
        int task_kl = blockIdx.y * bsizey + tile * threadsy + ty;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n % nf3kl;
            dm_kl[k] = dm[kl_loc0+kl];
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
        for (int n = yslot_id; n < nf3ij; n += yslots) {
            vj_ij_cache[tx+n*threadsx] = 0;
        }
        for (int batch_kl = 0; batch_kl < tiley; ++batch_kl) {
            int task_kl0 = blockIdx.y * bsizey + batch_kl * threadsy;
            if (task_kl0 >= npairs_kl) {
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
            __syncthreads();
            // load dm_kl_cache from dm_kl regisers of each thread
            int addr0 = batch_kl * nf3kl;
            int addr1 = addr0 + nf3kl;
            switch (addr0 / xslots) {
            case 0:
                for (int n = 0; n <  3; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 1:
                for (int n = 1; n <= 4; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 2:
                for (int n = 2; n <= 5; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 3:
                for (int n = 3; n <= 6; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 4:
                for (int n = 4; n <= 7; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 5:
                for (int n = 5; n <= 8; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 6:
                for (int n = 6; n <= 9; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 7:
                for (int n = 7; n <= 10; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 8:
                for (int n = 8; n <= 11; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 9:
                for (int n = 9; n <= 12; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 10:
                for (int n = 10; n <= 13; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 11:
                for (int n = 11; n <= 14; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 12:
                for (int n = 12; n <= 15; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 13:
                for (int n = 13; n <= 16; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 14:
                for (int n = 14; n <= 17; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 15:
                for (int n = 15; n <= 18; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 16:
                for (int n = 16; n <= 19; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 17:
                for (int n = 17; n <= 20; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 18:
                for (int n = 18; n <= 21; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 19:
                for (int n = 19; n <= 22; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 20:
                for (int n = 20; n <= 23; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 21:
                for (int n = 21; n <= 24; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 22:
                for (int n = 22; n <= 25; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 23:
                for (int n = 23; n <= 26; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 24:
                for (int n = 24; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 25:
                for (int n = 25; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 26:
                for (int n = 26; n <= 27; ++n) {
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            case 27:
                {
                    int n = 27;
                    int addr = n * xslots + xslot_id;
                    if (addr0 <= addr && addr < addr1) {
                        dm_kl_cache[ty+(addr-addr0)*threadsy] = dm_kl[n];
                    }
                }
                break;
            default:
                for (int n = xslot_id; n < nf3kl; n += xslots) {
                    // Assign a special value to dm cache. When the output shows
                    // values ~1e200, this indicates that tiley size is too big.
                    // j_engine scheme function should be adjusted.
                    dm_kl_cache[ty+n*threadsy] = -1e200;
                }
            }

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
                default:
                    iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsq_per_block, gout_id, gout_stride);
                }
            }

            Rt = Rt_buf;
            double *vj_cache = Rt + nf3ijkl * nsq_per_block;
            uint16_t *p1 = Rt2_ij_kl + Rt2_idx_offsets[lij*RT2_MAX+lkl];
            int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lkl];
            for (int i = gout_id; i < nf3ij+gout_id; i += gout_stride) {
                __syncthreads();
                double val = 0.;
                if (i < nf3ij) {
                    int off = i * nf3kl;
                    for (int k = 0; k < nf3kl; ++k) {
                        double s = Rt[sq_id+p1[off+k]*nsq_per_block];
                        val += efg_phase[k] * s * dm_kl_cache[ty+k*threadsy];
                    }
                }
                vj_cache[t_id] = val;
                for (int stride = threadsy/2; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache[t_id] += vj_cache[t_id + stride*threadsx];
                    }
                }
                __syncthreads();
                if (ty == 0 && i < nf3ij) {
                    vj_ij_cache[tx+i*threadsx] += vj_cache[t_id];
                }
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_ij_loc[task_ij];
            for (int n = yslot_id; n < nf3ij; n += yslots) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*threadsx]);
            }
        }
    }
}
