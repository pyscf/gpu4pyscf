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

#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks.cu"

__global__
void rys_j_kernel_experiment(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int lane_id = t_id % 32;
    int group_id = lane_id / threadsx;
    unsigned int mask = ((1 << threadsx) - 1) << group_id * threadsx;
    int tx = sq_id % threadsx;
    int ty = sq_id / threadsx;
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
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;

    extern __shared__ double dm_ij_cache[];
    double *dm_kl_cache = dm_ij_cache + nf3ij * threadsx;
    double *vj_ij_cache = dm_kl_cache + nf3kl * threadsy;
    double *vj_kl_cache = vj_ij_cache + nf3ij * threadsx;
    double *rjri = vj_kl_cache + nf3kl * threadsy;
    double *rlrk = rjri + threadsx * 3;
    double *cicj_cache = rlrk + threadsy * 3;
    double *Rpq = cicj_cache + threadsx * iprim*jprim + sq_id;
    double *rw = Rpq + nsq_per_block * 3;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *buf1 = gz + nsq_per_block * g_size;
    double *buf2 = buf1 + ((lij+1)*(lkl+1)*(max(lij,lkl)+2)/2)*nsq_per_block;
    double *vj_cache = buf1 - sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int batch_ij = 0; batch_ij < tilex; ++batch_ij) {
        int task_ij0 = blockIdx.x * bsizex + batch_ij * threadsx;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        int task_ij = task_ij0 + tx;
        double fac_sym = PI_FAC;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
            fac_sym = 0.;
        }
        int pair_ij = pair_ij_mapping[task_ij];
        int ish = pair_ij / nbas;
        int jsh = pair_ij % nbas;
        if (ish == jsh) fac_sym *= .5;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        __syncthreads();
        if (gout_id == 0 && ty == 0) {
            rjri[tx+0*threadsx] = xjxi;
            rjri[tx+1*threadsx] = yjyi;
            rjri[tx+2*threadsx] = zjzi;
        }
        int slot_id = gout_id * threadsy + ty;
        int slots = gout_stride * threadsy;
        for (int ij = slot_id; ij < iprim*jprim; ij += slots) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*threadsx] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = slot_id; n < nf3ij; n += slots) {
                dm_ij_cache[tx+n*threadsx] = dm[ij_loc0+n];
                vj_ij_cache[tx+n*threadsx] = 0;
            }
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

            int task_kl = task_kl0 + ty;
            double fac_sym = 1.;
            if (task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac_sym = 0.;
            }
            int pair_kl = pair_kl_mapping[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac_sym *= .5;
            if (pair_ij_mapping == pair_kl_mapping) {
                if (task_ij == task_kl) fac_sym *= .5;
                // TODO: skip certain blocks when task_ij < task_kl
                if (task_ij < task_kl) fac_sym = 0.;
            }
            __syncthreads();
            int slots = gout_stride * threadsx;
            int slot_id = gout_id * threadsx + tx;
            int kl_loc0 = pair_loc[pair_kl];
            for (int n = slot_id; n < nf3kl; n += slots) {
                dm_kl_cache[ty+n*threadsy] = dm[kl_loc0+n];
                vj_kl_cache[ty+n*threadsy] = 0;
            }

            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0 && tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0*threadsy] = xlxk;
                rlrk[ty+1*threadsy] = ylyk;
                rlrk[ty+2*threadsy] = zlzk;
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
                    double xlxk = rlrk[ty+0*threadsy];
                    double ylyk = rlrk[ty+1*threadsy];
                    double zlzk = rlrk[ty+2*threadsy];
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    gx[0] = fac_sym * ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0*threadsx] * aj_aij;
                    double yij = ri[1] + rjri[tx+1*threadsx] * aj_aij;
                    double zij = ri[2] + rjri[tx+2*threadsx] * aj_aij;
                    double xkl = rk[0] + rlrk[ty+0*threadsy] * al_akl;
                    double ykl = rk[1] + rlrk[ty+1*threadsy] * al_akl;
                    double zkl = rk[2] + rlrk[ty+2*threadsy] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    __syncthreads();
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    if (gout_id == 0) {
                        double cicj = cicj_cache[tx+ijp*threadsx];
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
                                double Rpa = rjri[tx+n*threadsx] * aj_aij;
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
                                double Rqc = rlrk[ty+_ix*threadsy] * al_akl;
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
                        //                dm_ij_cache[(i3xy+iz)*nsq_per_block];
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
                                    dm_ij_cache[tx+(i3xy+iz)*threadsx];
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
                        for (int jxyz = gout_id; jxyz < nf3kl+gout_id; jxyz+=gout_stride) {
                            double val = 0;
                            if (jxyz < nf3kl) {
                                Fold3Index f3i = kl_fold3idx[jxyz];
                                int kx = f3i.x;
                                int jyz = f3i.fold2yz;
                                for (int ix = 0; ix <= lij; ++ix) {
                                    val += gx[(ix+stride_k*kx)*nsq_per_block] *
                                        buf2[(ix*nf_kl+jyz)*nsq_per_block];
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
                            //if (tx == 0 && task_kl0+ty < npairs_kl && jxyz < nf3kl) {
                            //    vj_kl_cache[ty+jxyz*threadsy] += vj_cache[t_id];
                            //}
                            for (int offset = threadsx/2; offset > 0; offset /= 2) {
                                val += __shfl_down_sync(mask, val, offset);
                            }
                            if (tx == 0 && jxyz < nf3kl) {
                                vj_kl_cache[ty+jxyz*threadsy] += val;
                            }
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
                                    dm_kl_cache[ty+(j3xy+kz)*threadsy];
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

                        //for (int ix = 0, ixyz = 0; ix <= lij; ++ix) {
                        //for (int iy = 0, iyz = 0; iy <= lij-ix; ++iy) { // TODO: fuse iy-iz loop
                        //for (int iz = 0; iz <= lij-ix-iy; ++iz, ++iyz, ++ixyz) {
                        //    double val = 0;
                        //    for (int kx = 0; kx <= lkl; ++kx) {
                        //        val += gx[(ix+stride_k*kx)*nsq_per_block] *
                        //            buf2[(kx*nf_ij+iyz)*nsq_per_block];
                        //    }
                        //    vj_ij_cache[ixyz*nsq_per_block] += val;
                        //} } }
                        for (int ixyz = gout_id; ixyz < nf3ij+gout_id; ixyz+=gout_stride) {
                            __syncthreads();
                            double val = 0;
                            if (ixyz < nf3ij) {
                                Fold3Index f3i = ij_fold3idx[ixyz];
                                int ix = f3i.x;
                                int iyz = f3i.fold2yz;
                                for (int kx = 0; kx <= lkl; ++kx) {
                                    val += gx[(ix+stride_k*kx)*nsq_per_block] *
                                        buf2[(kx*nf_ij+iyz)*nsq_per_block];
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
                            if (ty == 0 && task_ij0+tx < npairs_ij && ixyz < nf3ij) {
                                vj_ij_cache[tx+ixyz*threadsx] += vj_cache[t_id];
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (task_kl0+ty < npairs_kl) {
                for (int n = slot_id; n < nf3kl; n += slots) {
                    atomicAdd(vj+kl_loc0+n, vj_kl_cache[ty+n*threadsy]);
                }
            }
        }
        // The last tile for ij
        if (task_ij0 + tx < npairs_ij) {
            int slots = gout_stride * threadsy;
            int slot_id = gout_id * threadsy + ty;
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = slot_id; n < nf3ij; n += slots) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*threadsx]);
            }
        }
    }
}

extern int rys_j_unrolled_experiment(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds);

extern "C" {
int RYS_build_j_experiment(double *vj, double *dm, int n_dm, int nao,
                RysIntEnvVars envs, int *scheme, int *shls_slice,
                int npairs_ij, int npairs_kl,
                int *pair_ij_mapping, int *pair_kl_mapping,
                float **qd_ij_max, float **qd_kl_max,
                float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = order / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    int lij = li + lj;
    int lkl = lk + ll;
    uint8_t stride_j = 1;
    uint8_t stride_k = lij + 1;
    uint8_t stride_l = lij + 1;
    int g_size = (lij + 1) * (lkl + 1);
    float *tile16_qd_ij_max = qd_ij_max[4];
    float *tile16_qd_kl_max = qd_kl_max[4];
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, tile16_qd_ij_max, s_estimator, tile16_qd_kl_max, cutoff};

    JKMatrix jk = {vj, NULL, dm, (uint16_t)n_dm};

    if (!rys_j_unrolled_experiment(&envs, &jk, &bounds)) {
        int threads_ij = scheme[0];
        int threads_kl = scheme[1];
        int gout_stride = scheme[2];
        int tilex = scheme[3];
        int tiley = scheme[4];
        switch (threads_ij) {
        case 1: bounds.qd_ij_max = qd_ij_max[0]; break;
        case 2: bounds.qd_ij_max = qd_ij_max[1]; break;
        case 4: bounds.qd_ij_max = qd_ij_max[2]; break;
        case 8: bounds.qd_ij_max = qd_ij_max[3]; break;
        case 16: bounds.qd_ij_max = qd_ij_max[4]; break;
        case 32: bounds.qd_ij_max = qd_ij_max[5]; break;
        }
        switch (threads_kl) {
        case 1: bounds.qd_kl_max = qd_kl_max[0]; break;
        case 2: bounds.qd_kl_max = qd_kl_max[1]; break;
        case 4: bounds.qd_kl_max = qd_kl_max[2]; break;
        case 8: bounds.qd_kl_max = qd_kl_max[3]; break;
        case 16: bounds.qd_kl_max = qd_kl_max[4]; break;
        case 32: bounds.qd_kl_max = qd_kl_max[5]; break;
        }
        int bsizex = threads_ij * tilex;
        int bsizey = threads_kl * tiley;
        int quartets_per_block = threads_ij * threads_kl;
        dim3 threads(quartets_per_block, gout_stride);
        int nmax = max(lij, lkl);
        int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
        int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
        int buflen = (nroots*2 + g_size*3 + 3) * quartets_per_block;
        buflen += iprim*jprim*threads_ij + 3 * (threads_ij + threads_kl);
        buflen += nf3ij * threads_ij * 2 + nf3kl * threads_kl * 2;
        buflen += (lij+1)*(lkl+1)*(nmax+2) * quartets_per_block;
        int blocks_ij = (npairs_ij + bsizex - 1) / bsizex;
        int blocks_kl = (npairs_kl + bsizey - 1) / bsizey;
        dim3 blocks(blocks_ij, blocks_kl);
        rys_j_kernel_experiment<<<blocks, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, threads_ij, threads_kl, tilex, tiley);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        printf("CUDA Error in RYS_build_j, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stdout);
        fprintf(stderr, "CUDA Error in RYS_build_j, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}
}
