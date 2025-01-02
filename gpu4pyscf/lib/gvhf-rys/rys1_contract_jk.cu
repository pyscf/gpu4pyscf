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

#include "vhf.cuh"
#include "rys1_roots.cu"
#include "create_tasks.cu"

#define GOUT_WIDTH      81

__device__
static void rys1_jk_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                            ShellQuartet *shl_quartet_idx, int ntasks,
                            double *hrr_cache)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int ijprim = iprim * jprim;
    int nprim = ijprim * kprim * lprim;
    int nroots = bounds.nroots;
    int np_rys = nprim * nroots;
    int lij = li + lj;
    int lkl = lk + ll;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int glen = nsq_per_block * g_size;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    double *env = envs.env;
    //double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double cicj_cache[];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        //int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cicj_cache[sq_id+ij*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }

        for (int prys = gout_id; prys < np_rys+gout_id; prys += gout_stride) {
            if (prys >= np_rys) {
                __syncthreads();
                continue;
            }
            int ijklp = prys % nprim;
            int klp = ijklp / ijprim;
            int ijp = ijklp % ijprim;

            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double xpa = xjxi * aj_aij; // (ai*xi+aj*xj)/aij
            double ypa = yjyi * aj_aij;
            double zpa = zjzi * aj_aij;
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            double xij = ri[0] + xpa;
            double yij = ri[1] + ypa;
            double zij = ri[2] + zpa;
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;

            double *gx = hrr_cache + prys * glen * 3 + sq_id;
            double *gy = gx + glen;
            double *gz = gy + glen;
            if (prys < nprim) {
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                double cicj = cicj_cache[sq_id+ijp*nsq_per_block];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                int glen3 = glen * 3 * nprim;
                for (int i = 0; i < nroots; ++i) {
                    gx[i*glen3] = fac;
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                double theta_rr = theta * rr;
                // write roots to gy, weights to gz
                rys_roots(nroots, theta_rr, gy, gz, glen3);
            }
            __syncthreads();
            double rt = gy[0];
            gy[0] = 1.;
            double rt_aa = rt / (aij + akl);

            // TRR
            //for i in range(lij):
            //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
            //for k in range(lkl):
            //    for i in range(lij+1):
            //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
            if (lij > 0) {
                double rt_aij = rt_aa * akl;
                double b10 = .5/aij * (1 - rt_aij);
                // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                double c0x = xpa - rt_aij * xpq;
                double c0y = ypa - rt_aij * ypq;
                double c0z = zpa - rt_aij * zpq;
                double s0x = gx[0];
                double s0y = gy[0];
                double s0z = gz[0];
                double s1x = c0x * s0x;
                double s1y = c0y * s0y;
                double s1z = c0z * s0z;
                gx[nsq_per_block] = s1x;
                gy[nsq_per_block] = s1y;
                gz[nsq_per_block] = s1z;
                for (int i = 1; i < lij; ++i) {
                    double ib = i * b10;
                    double s2x = c0x * s1x + ib * s0x;
                    double s2y = c0y * s1y + ib * s0y;
                    double s2z = c0z * s1z + ib * s0z;
                    gx[(i+1)*nsq_per_block] = s2x;
                    gy[(i+1)*nsq_per_block] = s2y;
                    gz[(i+1)*nsq_per_block] = s2z;
                    s0x = s1x;
                    s0y = s1y;
                    s0z = s1z;
                    s1x = s2x;
                    s1y = s2y;
                    s1z = s2z;
                }
            }

            if (lkl > 0) {
                double rt_akl = rt_aa * aij;
                double b00 = .5 * rt_aa;
                double b01 = .5/akl * (1 - rt_akl);
                for (int i = 0; i <= lij; ++i) {
                    double cpx = xqc + rt_akl * xpq;
                    double cpy = yqc + rt_akl * ypq;
                    double cpz = zqc + rt_akl * zpq;
                    //for i in range(lij+1):
                    //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                    double s0x = gx[i*nsq_per_block];
                    double s0y = gy[i*nsq_per_block];
                    double s0z = gz[i*nsq_per_block];
                    double s1x = cpx * s0x;
                    double s1y = cpy * s0y;
                    double s1z = cpz * s0z;
                    double ib;
                    if (i > 0) {
                        ib = i * b00;
                        s1x += ib * gx[(i-1)*nsq_per_block];
                        s1y += ib * gy[(i-1)*nsq_per_block];
                        s1z += ib * gz[(i-1)*nsq_per_block];
                    }
                    gx[(stride_k+i)*nsq_per_block] = s1x;
                    gy[(stride_k+i)*nsq_per_block] = s1y;
                    gz[(stride_k+i)*nsq_per_block] = s1z;

                    //for k in range(1, lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    for (int k = 1; k < lkl; ++k) {
                        double kb = k * b01;
                        double s2x = cpx*s1x + kb*s0x;
                        double s2y = cpy*s1y + kb*s0y;
                        double s2z = cpz*s1z + kb*s0z;
                        if (i > 0) {
                            s2x += ib * gx[(k*stride_k+i-1)*nsq_per_block];
                            s2y += ib * gy[(k*stride_k+i-1)*nsq_per_block];
                            s2z += ib * gz[(k*stride_k+i-1)*nsq_per_block];
                        }
                        gx[(k*stride_k+stride_k+i)*nsq_per_block] = s2x;
                        gy[(k*stride_k+stride_k+i)*nsq_per_block] = s2y;
                        gz[(k*stride_k+stride_k+i)*nsq_per_block] = s2z;
                        s0x = s1x;
                        s0y = s1y;
                        s0z = s1z;
                        s1x = s2x;
                        s1y = s2y;
                        s1z = s2z;
                    }
                }
            }

            // hrr
            // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
            // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
            if (lj > 0) {
                for (int m = 0; m <= lkl; ++m) {
                    int k = m * stride_k * nsq_per_block;
                    for (int j = 0; j < lj; ++j) {
                        int ij = (lij-j) + j*stride_j;
                        double s1x = gx[k+ij*nsq_per_block];
                        double s1y = gy[k+ij*nsq_per_block];
                        double s1z = gz[k+ij*nsq_per_block];
                        for (--ij; ij >= j*stride_j; --ij) {
                            double s0x = gx[k+ij*nsq_per_block];
                            double s0y = gy[k+ij*nsq_per_block];
                            double s0z = gz[k+ij*nsq_per_block];
                            gx[k+(ij+stride_j)*nsq_per_block] = s1x - xjxi * s0x;
                            gy[k+(ij+stride_j)*nsq_per_block] = s1y - yjyi * s0y;
                            gz[k+(ij+stride_j)*nsq_per_block] = s1z - zjzi * s0z;
                            s1x = s0x;
                            s1y = s0y;
                            s1z = s0z;
                        }
                    }
                }
            }
            if (ll > 0) {
                for (int n = 0; n < stride_k; ++n) {
                    int ij = n * nsq_per_block;
                    for (int l = 0; l < ll; ++l) {
                        int kl = (lkl-l)*stride_k + l*stride_l;
                        double s1x = gx[ij+kl*nsq_per_block];
                        double s1y = gy[ij+kl*nsq_per_block];
                        double s1z = gz[ij+kl*nsq_per_block];
                        for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) {
                            double s0x = gx[ij+kl*nsq_per_block];
                            double s0y = gy[ij+kl*nsq_per_block];
                            double s0z = gz[ij+kl*nsq_per_block];
                            gx[ij+(kl+stride_l)*nsq_per_block] = s1x - xlxk * s0x;
                            gy[ij+(kl+stride_l)*nsq_per_block] = s1y - ylyk * s0y;
                            gz[ij+(kl+stride_l)*nsq_per_block] = s1z - zlzk * s0z;
                            s1x = s0x;
                            s1y = s0y;
                            s1z = s0z;
                        }
                    }
                }
            }
        }

        __syncthreads();
        int nfi = bounds.nfi;
        int nfk = bounds.nfk;
        int nfij = bounds.nfij;
        int nfkl = bounds.nfkl;
        int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
        int *idy_ij = idx_ij + nfij;
        int *idz_ij = idy_ij + nfij;
        int *idx_kl = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1+ll];
        int *idy_kl = idx_kl + nfkl;
        int *idz_kl = idy_kl + nfkl;
        double gout[GOUT_WIDTH];
        for (int gout_start = 0; gout_start < nfij*nfkl;
             gout_start+=gout_stride*GOUT_WIDTH) {
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

            for (int prys = 0; prys < np_rys; ++prys) {
                double *gx = hrr_cache + prys * glen * 3 + sq_id;
                double *gy = gx + glen;
                double *gz = gy + glen;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijkl = (gout_start + n*gout_stride+gout_id);
                    int kl = ijkl / nfij;
                    int ij = ijkl % nfij;
                    if (kl >= nfkl) break;
                    int addrx = (idx_ij[ij] + idx_kl[kl] * stride_k) * nsq_per_block;
                    int addry = (idy_ij[ij] + idy_kl[kl] * stride_k) * nsq_per_block;
                    int addrz = (idz_ij[ij] + idz_kl[kl] * stride_k) * nsq_per_block;
                    gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                }
            }

            if (task_id >= ntasks) {
                continue;
            }
            double *dm = jk.dm;
            double *vj = jk.vj;
            double *vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            int *ao_loc = envs.ao_loc;
            int i0 = ao_loc[ish];
            int j0 = ao_loc[jsh];
            int k0 = ao_loc[ksh];
            int l0 = ao_loc[lsh];
            int nao = ao_loc[nbas];
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ijkl = (gout_start + n*gout_stride+gout_id);
                    int kl = ijkl / nfij;
                    int ij = ijkl % nfij;
                    if (kl >= nfkl) break;
                    double s = gout[n];
                    int i = ij % nfi;
                    int j = ij / nfi;
                    int k = kl % nfk;
                    int l = kl / nfk;
                    int _i = i + i0;
                    int _j = j + j0;
                    int _k = k + k0;
                    int _l = l + l0;
                    if (do_j) {
                        int _ji = _j*nao+_i;
                        int _lk = _l*nao+_k;
                        atomicAdd(vj+_lk, s * dm[_ji]);
                        atomicAdd(vj+_ji, s * dm[_lk]);
                    }
                    if (do_k) {
                        int _jl = _j*nao+_l;
                        int _jk = _j*nao+_k;
                        int _il = _i*nao+_l;
                        int _ik = _i*nao+_k;
                        atomicAdd(vk+_ik, s * dm[_jl]);
                        atomicAdd(vk+_il, s * dm[_jk]);
                        atomicAdd(vk+_jk, s * dm[_il]);
                        atomicAdd(vk+_jl, s * dm[_ik]);
                    }
                }
                vj += nao * nao;
                vk += nao * nao;
                dm += nao * nao;
            }
        }
    }
}

__global__
void rys1_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                   ShellQuartet *pool, double *hrr_pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;

    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int np_rys = iprim * jprim * kprim * lprim * nroots;
    int g_size = bounds.stride_l * (bounds.ll + 1) * blockDim.x;
    double *hrr_cache = hrr_pool + b_id * np_rys * g_size * 3;

    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        int ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        if (ntasks > 0) {
            rys1_jk_general(envs, jk, bounds, shl_quartet_idx, ntasks, hrr_cache);
        }
        __syncthreads();
    }
}
