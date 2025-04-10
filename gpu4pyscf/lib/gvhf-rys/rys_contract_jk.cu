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
#include "rys_roots.cu"
#include "create_tasks.cu"

// TODO: benchmark performance for 34, 36, 41, 43, 45, 47, 51, 57
#define GOUT_WIDTH      42

__device__
static void rys_jk_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                           ShellQuartet *shl_quartet_idx, int ntasks)
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
    int nfi = bounds.nfi;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfk = bounds.nfk;
    int nfl = (ll + 1) * (ll + 2) / 2;
    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *idx_kl = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1+ll];
    int *idy_kl = idx_kl + nfkl;
    int *idz_kl = idy_kl + nfkl;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    //double *env = c_env;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];

    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sq_id;
    double *rlrk = rjri + nsq_per_block * 3;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *cicj_cache = Rpq + nsq_per_block * 3;
    double *rw = cicj_cache + nsq_per_block * iprim * jprim;
    double *g = rw + nsq_per_block * nroots * 2;
    double *gx = g;
    double *gy = g + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;

    double gout[GOUT_WIDTH];

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
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
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
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        if (gout_id == 0) {
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            rjri[0*nsq_per_block] = xjxi;
            rjri[1*nsq_per_block] = yjyi;
            rjri[2*nsq_per_block] = zjzi;
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[ij*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int gout_start = 0; gout_start < nfij*nfkl; gout_start+=gout_stride*GOUT_WIDTH) {
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            __syncthreads();
            if (gout_id == 0) {
                double xlxk = rlrk[0*nsq_per_block];
                double ylyk = rlrk[1*nsq_per_block];
                double zlzk = rlrk[2*nsq_per_block];
                double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * rr_kl);
                double ckcl = ck[kp] * cl[lp] * Kcd;
                gx[0] = ckcl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xij = ri[0] + rjri[0*nsq_per_block] * aj_aij;
                double yij = ri[1] + rjri[1*nsq_per_block] * aj_aij;
                double zij = ri[2] + rjri[2*nsq_per_block] * aj_aij;
                double xkl = rk[0] + rlrk[0*nsq_per_block] * al_akl;
                double ykl = rk[1] + rlrk[1*nsq_per_block] * al_akl;
                double zkl = rk[2] + rlrk[2*nsq_per_block] * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                __syncthreads();
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    double cicj = cicj_cache[ijp*nsq_per_block];
                    gy[0] = cicj / (aij*akl*sqrt(aij+akl));
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, gout_id, gout_stride);
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gz[0] = rw[(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[irys*2*nsq_per_block];
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
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = g + n * g_size * nsq_per_block;
                            double Rpa = rjri[n*nsq_per_block] * aj_aij;
                            double c0x = Rpa - rt_aij * Rpq[n*nsq_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nsq_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nsq_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    if (lkl > 0) {
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            double *_gx = g + (i + _ix * g_size) * nsq_per_block;
                            double Rqc = rlrk[_ix*nsq_per_block] * al_akl;
                            double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsq_per_block];
                                }
                                _gx[stride_k*nsq_per_block] = s1x;
                            }

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nsq_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }
                    }

                    // hrr
                    // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                    // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                    if (lj > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            int lkl3 = (lkl+1)*3;
                            for (int m = gout_id; m < lkl3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m % 3;
                                double xjxi = rjri[_ix*nsq_per_block];
                                double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                                for (int j = 0; j < lj; ++j) {
                                    int ij = (lij-j) + j*stride_j;
                                    s1x = _gx[ij*nsq_per_block];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[ij*nsq_per_block];
                                        _gx[(ij+stride_j)*nsq_per_block] = s1x - xjxi * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }
                    if (ll > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xlxk = rlrk[_ix*nsq_per_block];
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
                                    int kl = (lkl-l)*stride_k + l*stride_l;
                                    s1x = _gx[kl*nsq_per_block];
                                    for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) {
                                        s0x = _gx[kl*nsq_per_block];
                                        _gx[(kl+stride_l)*nsq_per_block] = s1x - xlxk * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }

                    __syncthreads();
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        int ijkl = gout_start + n*gout_stride+gout_id;
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
                        if (kl >= nfkl) break;
                        int addrx = (idx_ij[ij] + idx_kl[kl] * stride_k) * nsq_per_block;
                        int addry = (idy_ij[ij] + idy_kl[kl] * stride_k) * nsq_per_block;
                        int addrz = (idz_ij[ij] + idz_kl[kl] * stride_k) * nsq_per_block;
                        gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                    }
                }
            }
        }

        double *dm = jk.dm;
        double *vj = jk.vj;
        double *vk = jk.vk;
        const bool do_j = (vj != NULL) && (task_id < ntasks);
        const bool do_k = (vk != NULL) && (task_id < ntasks);

        double* j_cache = rw;
        double* j_ij = j_cache;
        double* j_kl = j_ij + nfij * nsq_per_block;
        double* k_cache = do_j ? (j_kl + nfkl * nsq_per_block) : j_cache;
        double* k_ik = k_cache;
        double* k_il = k_ik + nfi * nfk * nsq_per_block;
        double* k_jk = k_il + nfi * nfl * nsq_per_block;
        double* k_jl = k_jk + nfj * nfk * nsq_per_block;
        int jk_cache_size = 0;
        if (do_j) jk_cache_size += nfij + nfkl;
        if (do_k) jk_cache_size += nfi * nfk + nfi * nfl + nfj * nfk + nfj * nfl;

        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            __syncthreads();
            for (int i = gout_id; i < jk_cache_size; i += gout_stride)
                j_cache[i * nsq_per_block] = 0;
            __syncthreads();
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
                    const int _ji = _j*nao+_i;
                    const int _lk = _l*nao+_k;
                    atomicAdd(j_kl + kl * nsq_per_block, s * dm[_ji]);
                    atomicAdd(j_ij + ij * nsq_per_block, s * dm[_lk]);
                }
                if (do_k) {
                    // The order of ik,il,jk,jl is consistent with ij,kl
                    // which is (j * nfi + i) and (l * nfk + k)
                    const int ik = k * nfi + i;
                    const int il = l * nfi + i;
                    const int jk = k * nfj + j;
                    const int jl = l * nfj + j;
                    const int _jl = _j*nao+_l;
                    const int _jk = _j*nao+_k;
                    const int _il = _i*nao+_l;
                    const int _ik = _i*nao+_k;
                    atomicAdd(k_ik + ik * nsq_per_block, s * dm[_jl]);
                    atomicAdd(k_il + il * nsq_per_block, s * dm[_jk]);
                    atomicAdd(k_jk + jk * nsq_per_block, s * dm[_il]);
                    atomicAdd(k_jl + jl * nsq_per_block, s * dm[_ik]);
                }
            }
            __syncthreads();
            if (do_j) {
                for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                    const int i = ij % nfi;
                    const int j = ij / nfi;
                    const int _i = i + i0;
                    const int _j = j + j0;
                    const int _ji = _j*nao+_i;
                    atomicAdd(vj + _ji, j_ij[ij * nsq_per_block]);
                }
                for (int kl = gout_id; kl < nfkl; kl += gout_stride) {
                    const int k = kl % nfk;
                    const int l = kl / nfk;
                    const int _k = k + k0;
                    const int _l = l + l0;
                    const int _lk = _l*nao+_k;
                    atomicAdd(vj + _lk, j_kl[kl * nsq_per_block]);
                }
            }
            if (do_k) {
                for (int ik = gout_id; ik < nfi * nfk; ik += gout_stride) {
                    const int i = ik % nfi;
                    const int k = ik / nfi;
                    const int _i = i + i0;
                    const int _k = k + k0;
                    const int _ik = _i*nao+_k;
                    atomicAdd(vk + _ik, k_ik[ik * nsq_per_block]);
                }
                for (int il = gout_id; il < nfi * nfl; il += gout_stride) {
                    const int i = il % nfi;
                    const int l = il / nfi;
                    const int _i = i + i0;
                    const int _l = l + l0;
                    const int _il = _i*nao+_l;
                    atomicAdd(vk + _il, k_il[il * nsq_per_block]);
                }
                for (int jk = gout_id; jk < nfj * nfk; jk += gout_stride) {
                    const int j = jk % nfj;
                    const int k = jk / nfj;
                    const int _j = j + j0;
                    const int _k = k + k0;
                    const int _jk = _j*nao+_k;
                    atomicAdd(vk + _jk, k_jk[jk * nsq_per_block]);
                }
                for (int jl = gout_id; jl < nfj * nfl; jl += gout_stride) {
                    const int j = jl % nfj;
                    const int l = jl / nfj;
                    const int _j = j + j0;
                    const int _l = l + l0;
                    const int _jl = _j*nao+_l;
                    atomicAdd(vk + _jl, k_jl[jl * nsq_per_block]);
                }
            }
            __syncthreads();

            vj += nao * nao;
            vk += nao * nao;
            dm += nao * nao;
        }
    } }
}

__global__
void rys_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                   ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    extern __shared__ int batch_id[];
    if (t_id == 0) {
        batch_id[0] = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id[0] < nbatches) {
        int batch_ij = batch_id[0] / nbatches_kl;
        int batch_kl = batch_id[0] % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            rys_jk_general(envs, jk, bounds, shl_quartet_idx, ntasks);
            __syncthreads();
        }
        if (t_id == 0) {
            batch_id[0] = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
