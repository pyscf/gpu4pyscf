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
#include <cuda.h>
#include <cuda_runtime.h>

#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks.cu"

__device__
static void rys_j_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                          ShellQuartet *shl_quartet_idx, int ntasks,
                          int ish0, int jsh0)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int lkl1 = lkl + 1;
    int nroots = bounds.nroots;
    int stride_k = bounds.stride_k;
    int g_size = stride_k * lkl1;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    double *dm = jk.dm;
    double *vj = jk.vj;
    int nf_ij = (lij+1)*(lij+2)/2;
    int nf_kl = (lkl+1)*(lkl+2)/2;
    int nf3ij = nf_ij*(lij+3)/3;
    int nf3kl = nf_kl*(lkl+3)/3;
    int ij_fold2idx_cum = lij*(lij+1)*(lij+2)/6;
    int kl_fold2idx_cum = lkl*(lkl+1)*(lkl+2)/6;
    int ij_fold3idx_cum = ij_fold2idx_cum*(lij+3)/4;
    int kl_fold3idx_cum = kl_fold2idx_cum*(lkl+3)/4;
    Fold2Index *ij_fold2idx = c_i_in_fold2idx + ij_fold2idx_cum;
    Fold2Index *kl_fold2idx = c_i_in_fold2idx + kl_fold2idx_cum;
    Fold3Index *ij_fold3idx = c_i_in_fold3idx + ij_fold3idx_cum;
    Fold3Index *kl_fold3idx = c_i_in_fold3idx + kl_fold3idx_cum;

    extern __shared__ double dm_ij_cache[];
    double *rw_cache = dm_ij_cache + nf3ij * TILE2;
    double *rw = rw_cache + sq_id;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *rjri = gz + nsq_per_block * g_size;
    double *rlrk = rjri + nsq_per_block * 3;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *cicj_cache = Rpq + nsq_per_block * 3;
    double *vj_ij = cicj_cache + iprim*jprim*nsq_per_block;
    double *dm_kl = vj_ij + nf3ij*nsq_per_block;
    double *vj_kl = dm_kl + nf3kl*nsq_per_block;
    double *buf1  = vj_kl + nf3kl*nsq_per_block;
    double *buf2  = buf1 + ((lij+1)*(lkl+1)*(MAX(lij,lkl)+2)/2)*nsq_per_block;

    for (int n = t_id; n < nf3ij*TILE2; n += threads) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
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
        double xlxk = rl[0] - rk[0];
        double ylyk = rl[1] - rk[1];
        double zlzk = rl[2] - rk[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        if (gout_id == 0) {
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

        for (int n = gout_id; n < nf3ij; n+=gout_stride) {
            vj_ij[n*nsq_per_block] = 0;
        }
        for (int n = gout_id; n < nf3kl; n+=gout_stride) {
            dm_kl[n*nsq_per_block] = dm[kl_pair0+n];
            vj_kl[n*nsq_per_block] = 0;
        }

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
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
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
                Rpq[0*nsq_per_block] = xpq;
                Rpq[1*nsq_per_block] = ypq;
                Rpq[2*nsq_per_block] = zpq;
                if (gout_id == 0) {
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
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
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
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3;
                            double *_gx = g + (i + _ix * g_size) * nsq_per_block;
                            double Rqc = rlrk[_ix*nsq_per_block] * al_akl;
                            double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsq_per_block];
                                }
                                _gx[stride_k*nsq_per_block] = s1x;
                            }

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

                    __syncthreads();
                    //for (int ix = 0, ixy = 0, i3xy = 0; ix <= lij; ++ix) {
                    //for (int iy = 0; iy <= lij-ix; ++iy, ++ixy) {
                    //    for (int kz = 0; kz <= lkl; ++kz) {
                    //        double val = 0;
                    //        for (int iz = 0; iz <= lij-ix-iy; ++iz) {
                    //            val += gz[(iz+stride_k*kz)*nsq_per_block] *
                    //                dm_ij[(i3xy+iz)*nsq_per_block];
                    //        }
                    //        buf1[(kz*nf_ij+ixy)*nsq_per_block] = val;
                    //    }
                    //    i3xy += lij-ix-iy;
                    //} }
                    for (int n = gout_id; n < nf_ij*(lkl+1); n+=gout_stride) {
                        int ixy = n % nf_ij;
                        int kz = n / nf_ij;
                        Fold2Index f2i = ij_fold2idx[ixy];
                        int ix = f2i.x;
                        int iy = f2i.y;
                        int i3xy = f2i.fold3offset;
                        double val = 0;
                        for (int iz = 0; iz <= lij-ix-iy; ++iz) {
                            val += gz[(iz+stride_k*kz)*nsq_per_block] *
                                dm_ij_cache[sh_ij+ (i3xy+iz)*TILE2];
                        }
                        buf1[n*nsq_per_block] = val;
                    }

                    __syncthreads();
                    //for (int ky = 0, jyz = 0; ky <= lkl; ++ky) {
                    //for (int kz = 0; kz <= lkl-ky; ++kz, ++jyz) {
                    //for (int ix = 0, ixy = 0; ix <= lij; ++ix) {
                    //    double val = 0;
                    //    for (int iy = 0; iy <= lij-ix; ++iy, ++ixy) {
                    //        val += gy[(iy+stride_k*ky)*nsq_per_block] *
                    //             buf1[(kz*nf_ij+ixy)*nsq_per_block];
                    //    }
                    //    buf2[(ix*nf_kl+jyz)*nsq_per_block] = val;
                    //} } }
                    for (int n = gout_id; n < nf_kl*(lij+1); n+=gout_stride) {
                        int jyz = n % nf_kl;
                        int ix = n / nf_kl;
                        Fold2Index f2i = kl_fold2idx[jyz];
                        int ixy = nf_ij-(lij-ix+1)*(lij-ix+2)/2;
                        int ky = f2i.x;
                        int kz = f2i.y;
                        double val = 0;
                        for (int iy = 0; iy <= lij-ix; ++iy, ++ixy) {
                            val += gy[(iy+stride_k*ky)*nsq_per_block] *
                                 buf1[(kz*nf_ij+ixy)*nsq_per_block];
                        }
                        buf2[n*nsq_per_block] = val;
                    }

                    __syncthreads();
                    //for (int kx = 0, jxyz = 0; kx <= lkl; ++kx) {
                    //for (int ky = 0, jyz = 0; ky <= lkl-kx; ++ky) {
                    //for (int kz = 0; kz <= lkl-kx-ky; ++kz, ++jyz, ++jxyz) {
                    //    double val = 0;
                    //    for (int ix = 0; ix <= lij; ++ix) {
                    //        val += gx[(ix+stride_k*kx)*nsq_per_block] *
                    //            buf2[(ix*nf_kl+jyz)*nsq_per_block];
                    //    }
                    //    vj_kl[jxyz*nsq_per_block] += val;
                    //} } }
                    for (int jxyz = gout_id; jxyz < nf3kl; jxyz+=gout_stride) {
                        Fold3Index f3i = kl_fold3idx[jxyz];
                        int kx = f3i.x;
                        int jyz = f3i.fold2yz;
                        double val = 0;
                        for (int ix = 0; ix <= lij; ++ix) {
                            val += gx[(ix+stride_k*kx)*nsq_per_block] *
                                buf2[(ix*nf_kl+jyz)*nsq_per_block];
                        }
                        vj_kl[jxyz*nsq_per_block] += val;
                    }

                    //for (int kx = 0, jxy = 0, j3xy = 0; kx <= lkl; ++kx) {
                    //for (int ky = 0; ky <= lkl-kx; ++ky, ++jxy) {
                    //    for (int iz = 0; iz <= lij; ++iz) {
                    //        double val = 0;
                    //        for (int kz = 0; kz <= lkl-kx-ky; ++kz) {
                    //            val += gz[(iz+stride_k*kz)*nsq_per_block] *
                    //                dm_kl[(j3xy+kz)*nsq_per_block];
                    //        }
                    //        buf1[(iz*nf_kl+jxy)*nsq_per_block] = val;
                    //    }
                    //    j3xy += lkl-kx-ky;
                    //} }
                    for (int n = gout_id; n < nf_kl*(lij+1); n+=gout_stride) {
                        int jxy = n % nf_kl;
                        int iz = n / nf_kl;
                        Fold2Index f2i = kl_fold2idx[jxy];
                        int kx = f2i.x;
                        int ky = f2i.y;
                        int j3xy = f2i.fold3offset;
                        double val = 0;
                        for (int kz = 0; kz <= lkl-kx-ky; ++kz) {
                            val += gz[(iz+stride_k*kz)*nsq_per_block] *
                                dm_kl[(j3xy+kz)*nsq_per_block];
                        }
                        buf1[n*nsq_per_block] = val;
                    }

                    __syncthreads();
                    //for (int iy = 0, iyz = 0; iy <= lij; ++iy) {
                    //for (int iz = 0; iz <= lij-iy; ++iz, ++iyz) {
                    //for (int kx = 0, jxy = 0; kx <= lkl; ++kx) {
                    //    double val = 0;
                    //    for (int ky = 0; ky <= lkl-kx; ++ky, ++jxy) {
                    //        val += gy[(iy+stride_k*ky)*nsq_per_block] *
                    //             buf1[(iz*nf_kl+jxy)*nsq_per_block];
                    //    }
                    //    buf2[(kx*nf_ij+iyz)*nsq_per_block] = val;
                    //} } }
                    for (int n = gout_id; n < nf_ij*(lkl+1); n+=gout_stride) {
                        int iyz = n % nf_ij;
                        int kx = n / nf_ij;
                        Fold2Index f2i = ij_fold2idx[iyz];
                        int jxy = nf_kl-(lkl-kx+1)*(lkl-kx+2)/2;
                        int iy = f2i.x;
                        int iz = f2i.y;
                        double val = 0;
                        for (int ky = 0; ky <= lkl-kx; ++ky, ++jxy) {
                            val += gy[(iy+stride_k*ky)*nsq_per_block] *
                                 buf1[(iz*nf_kl+jxy)*nsq_per_block];
                        }
                        buf2[n*nsq_per_block] = val;
                    }

                    __syncthreads();
                    //for (int ix = 0, ixyz = 0; ix <= lij; ++ix) {
                    //for (int iy = 0, iyz = 0; iy <= lij-ix; ++iy) { // TODO: fuse iy-iz loop
                    //for (int iz = 0; iz <= lij-ix-iy; ++iz, ++iyz, ++ixyz) {
                    //    double val = 0;
                    //    for (int kx = 0; kx <= lkl; ++kx) {
                    //        val += gx[(ix+stride_k*kx)*nsq_per_block] *
                    //            buf2[(kx*nf_ij+iyz)*nsq_per_block];
                    //    }
                    //    vj_ij[ixyz*nsq_per_block] += val;
                    //} } }
                    for (int ixyz = gout_id; ixyz < nf3ij; ixyz+=gout_stride) {
                        Fold3Index f3i = ij_fold3idx[ixyz];
                        int ix = f3i.x;
                        int iyz = f3i.fold2yz;
                        double val = 0;
                        for (int kx = 0; kx <= lkl; ++kx) {
                            val += gx[(ix+stride_k*kx)*nsq_per_block] *
                                buf2[(kx*nf_ij+iyz)*nsq_per_block];
                        }
                        vj_ij[ixyz*nsq_per_block] += val;
                    }
                }
            }
        }
        __syncthreads();
        if (task_id >= ntasks) {
            continue;
        }
        for (int n = gout_id; n < nf3ij; n+=gout_stride) {
            atomicAdd(vj+ij_pair0+n, vj_ij[n*nsq_per_block]);
        }
        for (int n = gout_id; n < nf3kl; n+=gout_stride) {
            atomicAdd(vj+kl_pair0+n, vj_kl[n*nsq_per_block]);
        }
    }
}

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_j_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                  ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int nbas = envs.nbas;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
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
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            rys_j_general(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        __syncthreads();
    }
}

__device__
static void rys_j_with_gout(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int lkl1 = lkl + 1;
    int nroots = bounds.nroots;
    int stride_k = bounds.stride_k;
    int g_size = stride_k * lkl1;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao_pairs = pair_loc[nbas*nbas];
    double *env = envs.env;
    double omega = envs.env[PTR_RANGE_OMEGA];
    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
    int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
    int ij_fold3idx_cum = lij*nf3ij/4;
    int kl_fold3idx_cum = lkl*nf3kl/4;
    Fold3Index *ij_fold3idx = c_i_in_fold3idx + ij_fold3idx_cum;
    Fold3Index *kl_fold3idx = c_i_in_fold3idx + kl_fold3idx_cum;

    extern __shared__ double rw_cache[];
    double *rw = rw_cache + sq_id;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *rjri = gz + nsq_per_block * g_size;
    double *rlrk = rjri + nsq_per_block * 3;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *cicj_cache = Rpq + nsq_per_block * 3;
    double *gout = cicj_cache + iprim*jprim*nsq_per_block;

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
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
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
        double xlxk = rl[0] - rk[0];
        double ylyk = rl[1] - rk[1];
        double zlzk = rl[2] - rk[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        if (gout_id == 0) {
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

        for (int n = gout_id; n < nf3ij*nf3kl; n += gout_stride) {
            gout[n*nsq_per_block] = 0;
        }

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
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
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
                Rpq[0*nsq_per_block] = xpq;
                Rpq[1*nsq_per_block] = ypq;
                Rpq[2*nsq_per_block] = zpq;
                if (gout_id == 0) {
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
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
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
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3;
                            double *_gx = g + (i + _ix * g_size) * nsq_per_block;
                            double Rqc = rlrk[_ix*nsq_per_block] * al_akl;
                            double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsq_per_block];
                                }
                                _gx[stride_k*nsq_per_block] = s1x;
                            }

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

                    __syncthreads();
                    for (int n = gout_id; n < nf3ij*nf3kl; n+=gout_stride) {
                        int i = n % nf3ij;
                        int k = n / nf3ij;
                        Fold3Index f3i = ij_fold3idx[i];
                        Fold3Index f3k = kl_fold3idx[k];
                        int ix = f3i.x;
                        int iy = f3i.y;
                        int iz = f3i.z;
                        int kx = f3k.x;
                        int ky = f3k.y;
                        int kz = f3k.z;
                        gout[n*nsq_per_block] +=
                            gx[(ix+stride_k*kx)*nsq_per_block] *
                            gy[(iy+stride_k*ky)*nsq_per_block] *
                            gz[(iz+stride_k*kz)*nsq_per_block];
                    }
                }
            }
        }
        __syncthreads();
        if (task_id >= ntasks) {
            continue;
        }
        double *dm = jk.dm;
        double *vj = jk.vj;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            for (int k = gout_id; k < nf3kl; k += gout_stride) {
                double vj_kl = 0.;
                for (int i = 0; i < nf3ij; ++i) {
                    vj_kl += dm[ij_pair0+i] * gout[(i+k*nf3ij)*nsq_per_block];
                }
                atomicAdd(vj+kl_pair0+k, vj_kl);
            }
            for (int i = gout_id; i < nf3ij; i += gout_stride) {
                double vj_ij = 0.;
                for (int k = 0; k < nf3kl; ++k) {
                    vj_ij += dm[kl_pair0+k] * gout[(i+k*nf3ij)*nsq_per_block];
                }
                atomicAdd(vj+ij_pair0+i, vj_ij);
            }
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_j_with_gout_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                            ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
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
            rys_j_with_gout(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
