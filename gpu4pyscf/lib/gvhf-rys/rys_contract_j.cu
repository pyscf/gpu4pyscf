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

__constant__ Fold2Index c_i_in_fold2idx[165];
__constant__ Fold3Index c_i_in_fold3idx[495];

__global__ static
void rys_j_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, int *pool)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int smid = get_smid();
    int *bas_kl_idx = pool + smid * QUEUE_DEPTH;
    __shared__ int ntasks;
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
    if (jk.omega >= 0) {
        _fill_vj_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    } else {
        _fill_sr_vj_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }

    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int g_size = bounds.g_size;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double omega = jk.omega;
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
    double *cicj_cache = dm_ij_cache + nf3ij;
    double *rw_cache = cicj_cache + iprim*jprim;
    double *rw = rw_cache + sq_id;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *rlrk = gz + nsq_per_block * g_size;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *vj_ij = Rpq + nsq_per_block * 3;
    double *dm_kl = vj_ij + nf3ij*nsq_per_block;
    double *vj_kl = dm_kl + nf3kl*nsq_per_block;
    double *buf1  = vj_kl + nf3kl*nsq_per_block;
    double *buf2  = buf1 + ((lij+1)*(lkl+1)*(MAX(lij,lkl)+2)/2)*nsq_per_block;

    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ double *expi;
    __shared__ double *expj;
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    int ij_pair0 = pair_loc[bas_ij];
    for (int n = t_id; n < nf3ij; n += threads) {
        dm_ij_cache[n] = dm[ij_pair0+n];
    }

    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int nbas = envs.nbas;
        int *bas = envs.bas;
        double *env = envs.env;
        int iprim = bounds.iprim;
        int jprim = bounds.jprim;
        int kprim = bounds.kprim;
        int lprim = bounds.lprim;
        int stride_k = bounds.stride_k;
        int g_size = bounds.g_size;

        int bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ish == jsh) fac_sym *= .5;
            if (ksh == lsh) fac_sym *= .5;
            if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }

        for (int n = gout_id; n < nf3ij; n+=gout_stride) {
            vj_ij[n*nsq_per_block] = 0;
        }
        if (task_id < ntasks) {
            int kl_pair0 = pair_loc[bas_kl];
            for (int n = gout_id; n < nf3kl; n+=gout_stride) {
                dm_kl[n*nsq_per_block] = dm[kl_pair0+n];
                vj_kl[n*nsq_per_block] = 0;
            }
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
                gx[0] = fac_sym * ckcl;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                __syncthreads();
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xij = ri[0] + rjri[0] * aj_aij;
                double yij = ri[1] + rjri[1] * aj_aij;
                double zij = ri[2] + rjri[2] * aj_aij;
                double xkl = rk[0] + rlrk[0*nsq_per_block] * al_akl;
                double ykl = rk[1] + rlrk[1*nsq_per_block] * al_akl;
                double zkl = rk[2] + rlrk[2*nsq_per_block] * al_akl;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    double cicj = cicj_cache[ijp];
                    gy[0] = cicj / (aij*akl*sqrt(aij+akl));
                    aij_cache[0] = aij;
                    aij_cache[1] = aj_aij;
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
                    double aij = aij_cache[0];
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
                            double Rpa = rjri[n] * aij_cache[1];
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
                            val += gz[(iz+stride_k*kz)*nsq_per_block] * dm_ij_cache[i3xy+iz];
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
        int kl_pair0 = pair_loc[bas_kl];
        for (int n = gout_id; n < nf3kl; n+=gout_stride) {
            atomicAdd(vj+kl_pair0+n, vj_kl[n*nsq_per_block]);
        }
    }
}

__global__ static
void rys_j_with_gout_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, int *pool)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int smid = get_smid();
    int *bas_kl_idx = pool + smid * QUEUE_DEPTH;
    __shared__ int ntasks;
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
    if (jk.omega >= 0) {
        _fill_vj_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    } else {
        _fill_sr_vj_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }

    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int g_size = bounds.g_size;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao_pairs = pair_loc[nbas*nbas];
    double *env = envs.env;
    double omega = jk.omega;
    int nf3ij = (lij+1)*(lij+2)*(lij+3)/6;
    int nf3kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
    int ij_fold3idx_cum = lij*nf3ij/4;
    int kl_fold3idx_cum = lkl*nf3kl/4;
    Fold3Index *ij_fold3idx = c_i_in_fold3idx + ij_fold3idx_cum;
    Fold3Index *kl_fold3idx = c_i_in_fold3idx + kl_fold3idx_cum;

    extern __shared__ double rw_cache[];
    double *cicj_cache = rw_cache;
    double *rw = cicj_cache + iprim*jprim + sq_id;
    double *g = rw + nsq_per_block * nroots*2;
    double *gx = g;
    double *gy = gx + nsq_per_block * g_size;
    double *gz = gy + nsq_per_block * g_size;
    double *rlrk = gz + nsq_per_block * g_size;
    double *Rpq = rlrk + nsq_per_block * 3;
    double *gout = Rpq + nsq_per_block * 3;

    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ double *expi;
    __shared__ double *expj;
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
        __syncthreads();
        int nbas = envs.nbas;
        int *bas = envs.bas;
        double *env = envs.env;
        int iprim = bounds.iprim;
        int jprim = bounds.jprim;
        int kprim = bounds.kprim;
        int lprim = bounds.lprim;
        int stride_k = bounds.stride_k;
        int g_size = bounds.g_size;

        int bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ish == jsh) fac_sym *= .5;
            if (ksh == lsh) fac_sym *= .5;
            if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }

        int ij_pair0 = pair_loc[bas_ij];
        int kl_pair0 = pair_loc[bas_kl];
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
                double xij = ri[0] + rjri[0] * aj_aij;
                double yij = ri[1] + rjri[1] * aj_aij;
                double zij = ri[2] + rjri[2] * aj_aij;
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
                    double cicj = cicj_cache[ijp];
                    gy[0] = cicj / (aij*akl*sqrt(aij+akl));
                    aij_cache[0] = aij;
                    aij_cache[1] = aj_aij;
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
                    double aij = aij_cache[0];
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
                            double Rpa = rjri[n] * aij_cache[1];
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

extern int rys_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds, int *pool);

extern "C" {
int RYS_build_j(double *vj, double *dm, int n_dm, int nao,
                RysIntEnvVars *envs, int *scheme, int *shls_slice,
                int npairs_ij, int npairs_kl, int *pair_ij_mapping, int *pair_kl_mapping,
                float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                int *pool, int *atm, int natm, int *bas, int nbas, double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4];
    int lsh0 = shls_slice[6];
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    int iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    int jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    int kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    int lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int nfl = (ll+1)*(ll+2)/2;
    int order = li + lj + lk + ll;
    int nroots = order / 2 + 1;
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
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, s_estimator, dm_cond, cutoff};

    JKMatrix jk = {vj, NULL, dm, n_dm, 0, omega};

    if (!rys_j_unrolled(envs, &jk, &bounds, pool)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int with_gout = scheme[2];
        dim3 threads(quartets_per_block, gout_stride);
        adjust_threads(rys_j_with_gout_kernel, threads.x);
        int nmax = MAX(lij, lkl);
        int nf3_ij = (lij+1)*(lij+2)*(lij+3)/6;
        int nf3_kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
        int buflen = (nroots*2 + g_size*3 + 6) * quartets_per_block + iprim*jprim;
        if (with_gout) {
            buflen += nf3_ij*nf3_kl * quartets_per_block;
            rys_j_with_gout_kernel<<<npairs_ij, threads, buflen*sizeof(double)>>>(*envs, jk, bounds, pool);
        } else {
            buflen += (nf3_ij+nf3_kl*2+(lij+1)*(lkl+1)*(nmax+2)) * quartets_per_block;
            buflen += nf3_ij; // dm_ij_cache
            rys_j_kernel<<<npairs_ij, threads, buflen*sizeof(double)>>>(*envs, jk, bounds, pool);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        fprintf(stderr, "CUDA Error in RYS_build_j, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_init_rysj_constant(int shm_size)
{
    Fold2Index i_in_fold2idx[165];
    Fold3Index i_in_fold3idx[495];
    int n2 = 0;
    int n3 = 0;
    for (int l = 0; l <= LMAX*2; ++l) {
        for (int i = 0, ijk = 0; i <= l; ++i) {
        for (int j = 0; j <= l-i; ++j, ++n2) {
            i_in_fold2idx[n2].x = i;
            i_in_fold2idx[n2].y = j;
            i_in_fold2idx[n2].fold3offset = ijk;
            for (int k = 0; k <= l-i-j; ++k, ++n3, ++ijk) {
                i_in_fold3idx[n3].x = i;
                i_in_fold3idx[n3].y = j;
                i_in_fold3idx[n3].z = k;
                i_in_fold3idx[n3].fold2yz = (l+1)*(l+2)/2 - (l-j+1)*(l-j+2)/2 + k;
            }
        } }
    }
    cudaMemcpyToSymbol(c_i_in_fold2idx, i_in_fold2idx, 165*sizeof(Fold2Index));
    cudaMemcpyToSymbol(c_i_in_fold3idx, i_in_fold3idx, 495*sizeof(Fold3Index));
    cudaFuncSetAttribute(rys_j_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_j_with_gout_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
