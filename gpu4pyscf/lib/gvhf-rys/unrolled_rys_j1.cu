#include <cuda.h>
#include <cuda_runtime.h>
#include "vhf.cuh"
#include "rys_roots.cu"


// TILEX=32, TILEY=32,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_j_0_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 512;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 512;
    double *dm_ij_cache = dm_kl_cache + 512;
    double *rjri = dm_ij_cache + 16;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 512; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 1; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_000 = 0;
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*32+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_000 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(1, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 1; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double dot_lij_z_000 = wt * dm_ij_cache[tx+0];
                        double dot_lij_y_000 = dot_lij_z_000;
                        vj_kl_000 += dot_lij_y_000;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = wt * dm_kl_cache[sq_kl+0];
                        double dot_lkl_y_000 = dot_lkl_z_000;
                        vj_ij_000 += dot_lkl_y_000;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_000 += __shfl_down_sync(mask, vj_kl_000, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+0] += vj_kl_000;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_000;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+0, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=32, TILEY=32,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_j_1_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 512;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 512;
    double *dm_ij_cache = dm_kl_cache + 512;
    double *rjri = dm_ij_cache + 64;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 512; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_001 = 0;
        double vj_ij_010 = 0;
        double vj_ij_100 = 0;
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*32+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_000 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(1, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 1; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double dot_lij_z_000 = trr_10z * dm_ij_cache[tx+16];
                        double dot_lij_z_010 = wt * dm_ij_cache[tx+32];
                        double dot_lij_z_100 = wt * dm_ij_cache[tx+48];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010;
                        double dot_lij_y_100 = dot_lij_z_100;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        vj_kl_000 += dot_lij_y_000 + trr_10x * dot_lij_y_100;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = wt * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_001 = trr_10z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_y_000 = dot_lkl_z_000;
                        double dot_lkl_y_001 = dot_lkl_z_001;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000;
                        vj_ij_001 += dot_lkl_y_001;
                        vj_ij_010 += dot_lkl_y_010;
                        vj_ij_100 += trr_10x * dot_lkl_y_000;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_000 += __shfl_down_sync(mask, vj_kl_000, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+0] += vj_kl_000;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_001;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+1, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_010;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_100;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=32, TILEY=11,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_j_1_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 176;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 704;
    double *dm_ij_cache = dm_kl_cache + 704;
    double *rjri = dm_ij_cache + 64;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 44; n += 16) {
        int i = n / 11;
        int tile = n % 11;
        int task_kl = blockIdx.y * 176 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*176] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 704; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_001 = 0;
        double vj_ij_010 = 0;
        double vj_ij_100 = 0;
        for (int batch_kl = 0; batch_kl < 11; ++batch_kl) {
            int task_kl0 = blockIdx.y * 176 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*11+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_010 = 0;
            double vj_kl_100 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(2, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double dot_lij_z_000 = trr_10z * dm_ij_cache[tx+16];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double dot_lij_z_001 = trr_11z * dm_ij_cache[tx+16];
                        double dot_lij_z_010 = wt * dm_ij_cache[tx+32];
                        double trr_01z = cpz * wt;
                        double dot_lij_z_011 = trr_01z * dm_ij_cache[tx+32];
                        double dot_lij_z_100 = wt * dm_ij_cache[tx+48];
                        double dot_lij_z_101 = trr_01z * dm_ij_cache[tx+48];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010;
                        double dot_lij_y_100 = dot_lij_z_100;
                        double dot_lij_y_101 = dot_lij_z_101;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+176];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+176];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+352];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+352];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+528];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+528];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010;
                        double dot_lkl_y_100 = dot_lkl_z_100;
                        double dot_lkl_y_101 = dot_lkl_z_101;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100;
                        vj_ij_001 += dot_lkl_y_001 + trr_01x * dot_lkl_y_101;
                        vj_ij_010 += dot_lkl_y_010 + trr_01x * dot_lkl_y_110;
                        vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+176] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+352] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+528] += vj_kl_100;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_001;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+1, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_010;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_100;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 44; n += 16) {
        int i = n / 11;
        int tile = n % 11;
        int task_kl = blockIdx.y * 176 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*176]);
        }
    }
}

// TILEX=32, TILEY=14,
__global__
void rys_j_1_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 224;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 2240;
    double *dm_ij_cache = dm_kl_cache + 2240;
    double *rjri = dm_ij_cache + 64;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 140; n += 16) {
        int i = n / 14;
        int tile = n % 14;
        int task_kl = blockIdx.y * 224 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*224] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 2240; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_001 = 0;
        double vj_ij_010 = 0;
        double vj_ij_100 = 0;
        for (int batch_kl = 0; batch_kl < 14; ++batch_kl) {
            int task_kl0 = blockIdx.y * 224 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*14+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_002 = 0;
            double vj_kl_010 = 0;
            double vj_kl_011 = 0;
            double vj_kl_020 = 0;
            double vj_kl_100 = 0;
            double vj_kl_101 = 0;
            double vj_kl_110 = 0;
            double vj_kl_200 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(2, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double dot_lij_z_000 = trr_10z * dm_ij_cache[tx+16];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double b01 = .5/akl_cache[ty] * (1 - rt_akl);
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double dot_lij_z_001 = trr_11z * dm_ij_cache[tx+16];
                        double trr_01z = cpz * wt;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double dot_lij_z_002 = trr_12z * dm_ij_cache[tx+16];
                        double dot_lij_z_010 = wt * dm_ij_cache[tx+32];
                        double dot_lij_z_011 = trr_01z * dm_ij_cache[tx+32];
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double dot_lij_z_012 = trr_02z * dm_ij_cache[tx+32];
                        double dot_lij_z_100 = wt * dm_ij_cache[tx+48];
                        double dot_lij_z_101 = trr_01z * dm_ij_cache[tx+48];
                        double dot_lij_z_102 = trr_02z * dm_ij_cache[tx+48];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011;
                        double dot_lij_y_002 = dot_lij_z_002 + trr_10y * dot_lij_z_012;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010;
                        double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010;
                        double dot_lij_y_100 = dot_lij_z_100;
                        double dot_lij_y_101 = dot_lij_z_101;
                        double dot_lij_y_102 = dot_lij_z_102;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100;
                        double dot_lij_y_111 = trr_01y * dot_lij_z_101;
                        double dot_lij_y_120 = trr_02y * dot_lij_z_100;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101;
                        vj_kl_002 += dot_lij_y_002 + trr_10x * dot_lij_y_102;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110;
                        vj_kl_011 += dot_lij_y_011 + trr_10x * dot_lij_y_111;
                        vj_kl_020 += dot_lij_y_020 + trr_10x * dot_lij_y_120;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100;
                        vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101;
                        vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+224] + trr_02z * dm_kl_cache[sq_kl+448];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+224] + trr_12z * dm_kl_cache[sq_kl+448];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+672] + trr_01z * dm_kl_cache[sq_kl+896];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+672] + trr_11z * dm_kl_cache[sq_kl+896];
                        double dot_lkl_z_020 = wt * dm_kl_cache[sq_kl+1120];
                        double dot_lkl_z_021 = trr_10z * dm_kl_cache[sq_kl+1120];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+1344] + trr_01z * dm_kl_cache[sq_kl+1568];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+1344] + trr_11z * dm_kl_cache[sq_kl+1568];
                        double dot_lkl_z_110 = wt * dm_kl_cache[sq_kl+1792];
                        double dot_lkl_z_111 = trr_10z * dm_kl_cache[sq_kl+1792];
                        double dot_lkl_z_200 = wt * dm_kl_cache[sq_kl+2016];
                        double dot_lkl_z_201 = trr_10z * dm_kl_cache[sq_kl+2016];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020;
                        double dot_lkl_y_100 = dot_lkl_z_100 + trr_01y * dot_lkl_z_110;
                        double dot_lkl_y_101 = dot_lkl_z_101 + trr_01y * dot_lkl_z_111;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110;
                        double dot_lkl_y_200 = dot_lkl_z_200;
                        double dot_lkl_y_201 = dot_lkl_z_201;
                        double dot_lkl_y_210 = trr_10y * dot_lkl_z_200;
                        vj_ij_001 += dot_lkl_y_001 + trr_01x * dot_lkl_y_101 + trr_02x * dot_lkl_y_201;
                        vj_ij_010 += dot_lkl_y_010 + trr_01x * dot_lkl_y_110 + trr_02x * dot_lkl_y_210;
                        vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100 + trr_12x * dot_lkl_y_200;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+224] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_002 += __shfl_down_sync(mask, vj_kl_002, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+448] += vj_kl_002;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+672] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_011 += __shfl_down_sync(mask, vj_kl_011, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+896] += vj_kl_011;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_020 += __shfl_down_sync(mask, vj_kl_020, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1120] += vj_kl_020;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1344] += vj_kl_100;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_101 += __shfl_down_sync(mask, vj_kl_101, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1568] += vj_kl_101;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_110 += __shfl_down_sync(mask, vj_kl_110, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1792] += vj_kl_110;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_200 += __shfl_down_sync(mask, vj_kl_200, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+2016] += vj_kl_200;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_001;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+1, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_010;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_100;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 140; n += 16) {
        int i = n / 14;
        int tile = n % 14;
        int task_kl = blockIdx.y * 224 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*224]);
        }
    }
}

// TILEX=32, TILEY=32,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_j_2_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 512;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 512;
    double *dm_ij_cache = dm_kl_cache + 512;
    double *rjri = dm_ij_cache + 160;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 512; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_001 = 0;
        double vj_ij_002 = 0;
        double vj_ij_010 = 0;
        double vj_ij_011 = 0;
        double vj_ij_020 = 0;
        double vj_ij_100 = 0;
        double vj_ij_101 = 0;
        double vj_ij_110 = 0;
        double vj_ij_200 = 0;
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*32+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_000 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(2, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double dot_lij_z_000 = trr_10z * dm_ij_cache[tx+16] + trr_20z * dm_ij_cache[tx+32];
                        double dot_lij_z_010 = wt * dm_ij_cache[tx+48] + trr_10z * dm_ij_cache[tx+64];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+80];
                        double dot_lij_z_100 = wt * dm_ij_cache[tx+96] + trr_10z * dm_ij_cache[tx+112];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+128];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+144];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110;
                        double dot_lij_y_200 = dot_lij_z_200;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        vj_kl_000 += dot_lij_y_000 + trr_10x * dot_lij_y_100 + trr_20x * dot_lij_y_200;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = wt * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_001 = trr_10z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_002 = trr_20z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_y_000 = dot_lkl_z_000;
                        double dot_lkl_y_001 = dot_lkl_z_001;
                        double dot_lkl_y_002 = dot_lkl_z_002;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000;
                        vj_ij_001 += dot_lkl_y_001;
                        vj_ij_002 += dot_lkl_y_002;
                        vj_ij_010 += dot_lkl_y_010;
                        vj_ij_011 += dot_lkl_y_011;
                        vj_ij_020 += dot_lkl_y_020;
                        vj_ij_100 += trr_10x * dot_lkl_y_000;
                        vj_ij_101 += trr_10x * dot_lkl_y_001;
                        vj_ij_110 += trr_10x * dot_lkl_y_010;
                        vj_ij_200 += trr_20x * dot_lkl_y_000;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_000 += __shfl_down_sync(mask, vj_kl_000, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+0] += vj_kl_000;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_001;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+1, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_010;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+4, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_100;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=32, TILEY=32,
__global__
void rys_j_2_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 512;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 2048;
    double *dm_ij_cache = dm_kl_cache + 2048;
    double *rjri = dm_ij_cache + 160;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 128; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 2048; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_001 = 0;
        double vj_ij_002 = 0;
        double vj_ij_010 = 0;
        double vj_ij_011 = 0;
        double vj_ij_020 = 0;
        double vj_ij_100 = 0;
        double vj_ij_101 = 0;
        double vj_ij_110 = 0;
        double vj_ij_200 = 0;
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*32+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_010 = 0;
            double vj_kl_100 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(2, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double dot_lij_z_000 = trr_10z * dm_ij_cache[tx+16] + trr_20z * dm_ij_cache[tx+32];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double dot_lij_z_001 = trr_11z * dm_ij_cache[tx+16] + trr_21z * dm_ij_cache[tx+32];
                        double dot_lij_z_010 = wt * dm_ij_cache[tx+48] + trr_10z * dm_ij_cache[tx+64];
                        double trr_01z = cpz * wt;
                        double dot_lij_z_011 = trr_01z * dm_ij_cache[tx+48] + trr_11z * dm_ij_cache[tx+64];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+80];
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+80];
                        double dot_lij_z_100 = wt * dm_ij_cache[tx+96] + trr_10z * dm_ij_cache[tx+112];
                        double dot_lij_z_101 = trr_01z * dm_ij_cache[tx+96] + trr_11z * dm_ij_cache[tx+112];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+128];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+128];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+144];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+144];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110;
                        double dot_lij_y_200 = dot_lij_z_200;
                        double dot_lij_y_201 = dot_lij_z_201;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+512];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+512];
                        double dot_lkl_z_002 = trr_21z * dm_kl_cache[sq_kl+512];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+1024];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+1024];
                        double dot_lkl_z_012 = trr_20z * dm_kl_cache[sq_kl+1024];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+1536];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+1536];
                        double dot_lkl_z_102 = trr_20z * dm_kl_cache[sq_kl+1536];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010;
                        double dot_lkl_y_100 = dot_lkl_z_100;
                        double dot_lkl_y_101 = dot_lkl_z_101;
                        double dot_lkl_y_102 = dot_lkl_z_102;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100;
                        vj_ij_001 += dot_lkl_y_001 + trr_01x * dot_lkl_y_101;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102;
                        vj_ij_010 += dot_lkl_y_010 + trr_01x * dot_lkl_y_110;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120;
                        vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+512] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1024] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1536] += vj_kl_100;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_001;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+1, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_010;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+4, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_100;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 128; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=32, TILEY=12,
__global__
void rys_j_2_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 192;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 1920;
    double *dm_ij_cache = dm_kl_cache + 1920;
    double *rjri = dm_ij_cache + 160;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 120; n += 16) {
        int i = n / 12;
        int tile = n % 12;
        int task_kl = blockIdx.y * 192 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*192] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 1920; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_001 = 0;
        double vj_ij_002 = 0;
        double vj_ij_010 = 0;
        double vj_ij_011 = 0;
        double vj_ij_020 = 0;
        double vj_ij_100 = 0;
        double vj_ij_101 = 0;
        double vj_ij_110 = 0;
        double vj_ij_200 = 0;
        for (int batch_kl = 0; batch_kl < 12; ++batch_kl) {
            int task_kl0 = blockIdx.y * 192 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*12+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_002 = 0;
            double vj_kl_010 = 0;
            double vj_kl_011 = 0;
            double vj_kl_020 = 0;
            double vj_kl_100 = 0;
            double vj_kl_101 = 0;
            double vj_kl_110 = 0;
            double vj_kl_200 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(3, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double dot_lij_z_000 = trr_10z * dm_ij_cache[tx+16] + trr_20z * dm_ij_cache[tx+32];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double b01 = .5/akl_cache[ty] * (1 - rt_akl);
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double dot_lij_z_001 = trr_11z * dm_ij_cache[tx+16] + trr_21z * dm_ij_cache[tx+32];
                        double trr_01z = cpz * wt;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double dot_lij_z_002 = trr_12z * dm_ij_cache[tx+16] + trr_22z * dm_ij_cache[tx+32];
                        double dot_lij_z_010 = wt * dm_ij_cache[tx+48] + trr_10z * dm_ij_cache[tx+64];
                        double dot_lij_z_011 = trr_01z * dm_ij_cache[tx+48] + trr_11z * dm_ij_cache[tx+64];
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double dot_lij_z_012 = trr_02z * dm_ij_cache[tx+48] + trr_12z * dm_ij_cache[tx+64];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+80];
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+80];
                        double dot_lij_z_022 = trr_02z * dm_ij_cache[tx+80];
                        double dot_lij_z_100 = wt * dm_ij_cache[tx+96] + trr_10z * dm_ij_cache[tx+112];
                        double dot_lij_z_101 = trr_01z * dm_ij_cache[tx+96] + trr_11z * dm_ij_cache[tx+112];
                        double dot_lij_z_102 = trr_02z * dm_ij_cache[tx+96] + trr_12z * dm_ij_cache[tx+112];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+128];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+128];
                        double dot_lij_z_112 = trr_02z * dm_ij_cache[tx+128];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+144];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+144];
                        double dot_lij_z_202 = trr_02z * dm_ij_cache[tx+144];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021;
                        double dot_lij_y_002 = dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020;
                        double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111;
                        double dot_lij_y_102 = dot_lij_z_102 + trr_10y * dot_lij_z_112;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110;
                        double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111;
                        double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110;
                        double dot_lij_y_200 = dot_lij_z_200;
                        double dot_lij_y_201 = dot_lij_z_201;
                        double dot_lij_y_202 = dot_lij_z_202;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200;
                        double dot_lij_y_211 = trr_01y * dot_lij_z_201;
                        double dot_lij_y_220 = trr_02y * dot_lij_z_200;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201;
                        vj_kl_002 += dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210;
                        vj_kl_011 += dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211;
                        vj_kl_020 += dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200;
                        vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201;
                        vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+192] + trr_02z * dm_kl_cache[sq_kl+384];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+192] + trr_12z * dm_kl_cache[sq_kl+384];
                        double dot_lkl_z_002 = trr_21z * dm_kl_cache[sq_kl+192] + trr_22z * dm_kl_cache[sq_kl+384];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+576] + trr_01z * dm_kl_cache[sq_kl+768];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+576] + trr_11z * dm_kl_cache[sq_kl+768];
                        double dot_lkl_z_012 = trr_20z * dm_kl_cache[sq_kl+576] + trr_21z * dm_kl_cache[sq_kl+768];
                        double dot_lkl_z_020 = wt * dm_kl_cache[sq_kl+960];
                        double dot_lkl_z_021 = trr_10z * dm_kl_cache[sq_kl+960];
                        double dot_lkl_z_022 = trr_20z * dm_kl_cache[sq_kl+960];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+1152] + trr_01z * dm_kl_cache[sq_kl+1344];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+1152] + trr_11z * dm_kl_cache[sq_kl+1344];
                        double dot_lkl_z_102 = trr_20z * dm_kl_cache[sq_kl+1152] + trr_21z * dm_kl_cache[sq_kl+1344];
                        double dot_lkl_z_110 = wt * dm_kl_cache[sq_kl+1536];
                        double dot_lkl_z_111 = trr_10z * dm_kl_cache[sq_kl+1536];
                        double dot_lkl_z_112 = trr_20z * dm_kl_cache[sq_kl+1536];
                        double dot_lkl_z_200 = wt * dm_kl_cache[sq_kl+1728];
                        double dot_lkl_z_201 = trr_10z * dm_kl_cache[sq_kl+1728];
                        double dot_lkl_z_202 = trr_20z * dm_kl_cache[sq_kl+1728];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020;
                        double dot_lkl_y_100 = dot_lkl_z_100 + trr_01y * dot_lkl_z_110;
                        double dot_lkl_y_101 = dot_lkl_z_101 + trr_01y * dot_lkl_z_111;
                        double dot_lkl_y_102 = dot_lkl_z_102 + trr_01y * dot_lkl_z_112;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110;
                        double dot_lkl_y_200 = dot_lkl_z_200;
                        double dot_lkl_y_201 = dot_lkl_z_201;
                        double dot_lkl_y_202 = dot_lkl_z_202;
                        double dot_lkl_y_210 = trr_10y * dot_lkl_z_200;
                        double dot_lkl_y_211 = trr_10y * dot_lkl_z_201;
                        double dot_lkl_y_220 = trr_20y * dot_lkl_z_200;
                        vj_ij_001 += dot_lkl_y_001 + trr_01x * dot_lkl_y_101 + trr_02x * dot_lkl_y_201;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202;
                        vj_ij_010 += dot_lkl_y_010 + trr_01x * dot_lkl_y_110 + trr_02x * dot_lkl_y_210;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220;
                        vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100 + trr_12x * dot_lkl_y_200;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+192] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_002 += __shfl_down_sync(mask, vj_kl_002, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+384] += vj_kl_002;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+576] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_011 += __shfl_down_sync(mask, vj_kl_011, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+768] += vj_kl_011;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_020 += __shfl_down_sync(mask, vj_kl_020, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+960] += vj_kl_020;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1152] += vj_kl_100;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_101 += __shfl_down_sync(mask, vj_kl_101, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1344] += vj_kl_101;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_110 += __shfl_down_sync(mask, vj_kl_110, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1536] += vj_kl_110;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_200 += __shfl_down_sync(mask, vj_kl_200, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1728] += vj_kl_200;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_001;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+1, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_010;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+4, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_100;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 120; n += 16) {
        int i = n / 12;
        int tile = n % 12;
        int task_kl = blockIdx.y * 192 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*192]);
        }
    }
}

// TILEX=32, TILEY=6,
__global__
void rys_j_2_3(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 96;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 1920;
    double *dm_ij_cache = dm_kl_cache + 1920;
    double *rjri = dm_ij_cache + 160;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 120; n += 16) {
        int i = n / 6;
        int tile = n % 6;
        int task_kl = blockIdx.y * 96 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*96] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 1920; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_001 = 0;
        double vj_ij_002 = 0;
        double vj_ij_010 = 0;
        double vj_ij_011 = 0;
        double vj_ij_020 = 0;
        double vj_ij_100 = 0;
        double vj_ij_101 = 0;
        double vj_ij_110 = 0;
        double vj_ij_200 = 0;
        for (int batch_kl = 0; batch_kl < 6; ++batch_kl) {
            int task_kl0 = blockIdx.y * 96 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*6+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_002 = 0;
            double vj_kl_003 = 0;
            double vj_kl_011 = 0;
            double vj_kl_012 = 0;
            double vj_kl_020 = 0;
            double vj_kl_021 = 0;
            double vj_kl_030 = 0;
            double vj_kl_101 = 0;
            double vj_kl_102 = 0;
            double vj_kl_110 = 0;
            double vj_kl_111 = 0;
            double vj_kl_120 = 0;
            double vj_kl_200 = 0;
            double vj_kl_201 = 0;
            double vj_kl_210 = 0;
            double vj_kl_300 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(3, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double dot_lij_z_000 = trr_10z * dm_ij_cache[tx+16] + trr_20z * dm_ij_cache[tx+32];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double b01 = .5/akl_cache[ty] * (1 - rt_akl);
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double dot_lij_z_001 = trr_11z * dm_ij_cache[tx+16] + trr_21z * dm_ij_cache[tx+32];
                        double trr_01z = cpz * wt;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double dot_lij_z_002 = trr_12z * dm_ij_cache[tx+16] + trr_22z * dm_ij_cache[tx+32];
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                        double dot_lij_z_003 = trr_13z * dm_ij_cache[tx+16] + trr_23z * dm_ij_cache[tx+32];
                        double dot_lij_z_010 = wt * dm_ij_cache[tx+48] + trr_10z * dm_ij_cache[tx+64];
                        double dot_lij_z_011 = trr_01z * dm_ij_cache[tx+48] + trr_11z * dm_ij_cache[tx+64];
                        double dot_lij_z_012 = trr_02z * dm_ij_cache[tx+48] + trr_12z * dm_ij_cache[tx+64];
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        double dot_lij_z_013 = trr_03z * dm_ij_cache[tx+48] + trr_13z * dm_ij_cache[tx+64];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+80];
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+80];
                        double dot_lij_z_022 = trr_02z * dm_ij_cache[tx+80];
                        double dot_lij_z_023 = trr_03z * dm_ij_cache[tx+80];
                        double dot_lij_z_100 = wt * dm_ij_cache[tx+96] + trr_10z * dm_ij_cache[tx+112];
                        double dot_lij_z_101 = trr_01z * dm_ij_cache[tx+96] + trr_11z * dm_ij_cache[tx+112];
                        double dot_lij_z_102 = trr_02z * dm_ij_cache[tx+96] + trr_12z * dm_ij_cache[tx+112];
                        double dot_lij_z_103 = trr_03z * dm_ij_cache[tx+96] + trr_13z * dm_ij_cache[tx+112];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+128];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+128];
                        double dot_lij_z_112 = trr_02z * dm_ij_cache[tx+128];
                        double dot_lij_z_113 = trr_03z * dm_ij_cache[tx+128];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+144];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+144];
                        double dot_lij_z_202 = trr_02z * dm_ij_cache[tx+144];
                        double dot_lij_z_203 = trr_03z * dm_ij_cache[tx+144];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021;
                        double dot_lij_y_002 = dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022;
                        double dot_lij_y_003 = dot_lij_z_003 + trr_10y * dot_lij_z_013 + trr_20y * dot_lij_z_023;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020;
                        double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021;
                        double dot_lij_y_012 = trr_01y * dot_lij_z_002 + trr_11y * dot_lij_z_012 + trr_21y * dot_lij_z_022;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020;
                        double dot_lij_y_021 = trr_02y * dot_lij_z_001 + trr_12y * dot_lij_z_011 + trr_22y * dot_lij_z_021;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                        double dot_lij_y_030 = trr_03y * dot_lij_z_000 + trr_13y * dot_lij_z_010 + trr_23y * dot_lij_z_020;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111;
                        double dot_lij_y_102 = dot_lij_z_102 + trr_10y * dot_lij_z_112;
                        double dot_lij_y_103 = dot_lij_z_103 + trr_10y * dot_lij_z_113;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110;
                        double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111;
                        double dot_lij_y_112 = trr_01y * dot_lij_z_102 + trr_11y * dot_lij_z_112;
                        double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110;
                        double dot_lij_y_121 = trr_02y * dot_lij_z_101 + trr_12y * dot_lij_z_111;
                        double dot_lij_y_130 = trr_03y * dot_lij_z_100 + trr_13y * dot_lij_z_110;
                        double dot_lij_y_200 = dot_lij_z_200;
                        double dot_lij_y_201 = dot_lij_z_201;
                        double dot_lij_y_202 = dot_lij_z_202;
                        double dot_lij_y_203 = dot_lij_z_203;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200;
                        double dot_lij_y_211 = trr_01y * dot_lij_z_201;
                        double dot_lij_y_212 = trr_01y * dot_lij_z_202;
                        double dot_lij_y_220 = trr_02y * dot_lij_z_200;
                        double dot_lij_y_221 = trr_02y * dot_lij_z_201;
                        double dot_lij_y_230 = trr_03y * dot_lij_z_200;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        vj_kl_002 += dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202;
                        vj_kl_003 += dot_lij_y_003 + trr_10x * dot_lij_y_103 + trr_20x * dot_lij_y_203;
                        vj_kl_011 += dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211;
                        vj_kl_012 += dot_lij_y_012 + trr_10x * dot_lij_y_112 + trr_20x * dot_lij_y_212;
                        vj_kl_020 += dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220;
                        vj_kl_021 += dot_lij_y_021 + trr_10x * dot_lij_y_121 + trr_20x * dot_lij_y_221;
                        vj_kl_030 += dot_lij_y_030 + trr_10x * dot_lij_y_130 + trr_20x * dot_lij_y_230;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201;
                        vj_kl_102 += trr_01x * dot_lij_y_002 + trr_11x * dot_lij_y_102 + trr_21x * dot_lij_y_202;
                        vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210;
                        vj_kl_111 += trr_01x * dot_lij_y_011 + trr_11x * dot_lij_y_111 + trr_21x * dot_lij_y_211;
                        vj_kl_120 += trr_01x * dot_lij_y_020 + trr_11x * dot_lij_y_120 + trr_21x * dot_lij_y_220;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200;
                        vj_kl_201 += trr_02x * dot_lij_y_001 + trr_12x * dot_lij_y_101 + trr_22x * dot_lij_y_201;
                        vj_kl_210 += trr_02x * dot_lij_y_010 + trr_12x * dot_lij_y_110 + trr_22x * dot_lij_y_210;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                        vj_kl_300 += trr_03x * dot_lij_y_000 + trr_13x * dot_lij_y_100 + trr_23x * dot_lij_y_200;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_02z * dm_kl_cache[sq_kl+192] + trr_03z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_001 = trr_12z * dm_kl_cache[sq_kl+192] + trr_13z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_002 = trr_22z * dm_kl_cache[sq_kl+192] + trr_23z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_010 = trr_01z * dm_kl_cache[sq_kl+480] + trr_02z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_011 = trr_11z * dm_kl_cache[sq_kl+480] + trr_12z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_012 = trr_21z * dm_kl_cache[sq_kl+480] + trr_22z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_020 = wt * dm_kl_cache[sq_kl+672] + trr_01z * dm_kl_cache[sq_kl+768];
                        double dot_lkl_z_021 = trr_10z * dm_kl_cache[sq_kl+672] + trr_11z * dm_kl_cache[sq_kl+768];
                        double dot_lkl_z_022 = trr_20z * dm_kl_cache[sq_kl+672] + trr_21z * dm_kl_cache[sq_kl+768];
                        double dot_lkl_z_030 = wt * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_031 = trr_10z * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_032 = trr_20z * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_100 = trr_01z * dm_kl_cache[sq_kl+1056] + trr_02z * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_101 = trr_11z * dm_kl_cache[sq_kl+1056] + trr_12z * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_102 = trr_21z * dm_kl_cache[sq_kl+1056] + trr_22z * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_110 = wt * dm_kl_cache[sq_kl+1248] + trr_01z * dm_kl_cache[sq_kl+1344];
                        double dot_lkl_z_111 = trr_10z * dm_kl_cache[sq_kl+1248] + trr_11z * dm_kl_cache[sq_kl+1344];
                        double dot_lkl_z_112 = trr_20z * dm_kl_cache[sq_kl+1248] + trr_21z * dm_kl_cache[sq_kl+1344];
                        double dot_lkl_z_120 = wt * dm_kl_cache[sq_kl+1440];
                        double dot_lkl_z_121 = trr_10z * dm_kl_cache[sq_kl+1440];
                        double dot_lkl_z_122 = trr_20z * dm_kl_cache[sq_kl+1440];
                        double dot_lkl_z_200 = wt * dm_kl_cache[sq_kl+1536] + trr_01z * dm_kl_cache[sq_kl+1632];
                        double dot_lkl_z_201 = trr_10z * dm_kl_cache[sq_kl+1536] + trr_11z * dm_kl_cache[sq_kl+1632];
                        double dot_lkl_z_202 = trr_20z * dm_kl_cache[sq_kl+1536] + trr_21z * dm_kl_cache[sq_kl+1632];
                        double dot_lkl_z_210 = wt * dm_kl_cache[sq_kl+1728];
                        double dot_lkl_z_211 = trr_10z * dm_kl_cache[sq_kl+1728];
                        double dot_lkl_z_212 = trr_20z * dm_kl_cache[sq_kl+1728];
                        double dot_lkl_z_300 = wt * dm_kl_cache[sq_kl+1824];
                        double dot_lkl_z_301 = trr_10z * dm_kl_cache[sq_kl+1824];
                        double dot_lkl_z_302 = trr_20z * dm_kl_cache[sq_kl+1824];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020 + trr_03y * dot_lkl_z_030;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021 + trr_03y * dot_lkl_z_031;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022 + trr_03y * dot_lkl_z_032;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020 + trr_13y * dot_lkl_z_030;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021 + trr_13y * dot_lkl_z_031;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020 + trr_23y * dot_lkl_z_030;
                        double dot_lkl_y_100 = dot_lkl_z_100 + trr_01y * dot_lkl_z_110 + trr_02y * dot_lkl_z_120;
                        double dot_lkl_y_101 = dot_lkl_z_101 + trr_01y * dot_lkl_z_111 + trr_02y * dot_lkl_z_121;
                        double dot_lkl_y_102 = dot_lkl_z_102 + trr_01y * dot_lkl_z_112 + trr_02y * dot_lkl_z_122;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110 + trr_12y * dot_lkl_z_120;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111 + trr_12y * dot_lkl_z_121;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110 + trr_22y * dot_lkl_z_120;
                        double dot_lkl_y_200 = dot_lkl_z_200 + trr_01y * dot_lkl_z_210;
                        double dot_lkl_y_201 = dot_lkl_z_201 + trr_01y * dot_lkl_z_211;
                        double dot_lkl_y_202 = dot_lkl_z_202 + trr_01y * dot_lkl_z_212;
                        double dot_lkl_y_210 = trr_10y * dot_lkl_z_200 + trr_11y * dot_lkl_z_210;
                        double dot_lkl_y_211 = trr_10y * dot_lkl_z_201 + trr_11y * dot_lkl_z_211;
                        double dot_lkl_y_220 = trr_20y * dot_lkl_z_200 + trr_21y * dot_lkl_z_210;
                        double dot_lkl_y_300 = dot_lkl_z_300;
                        double dot_lkl_y_301 = dot_lkl_z_301;
                        double dot_lkl_y_302 = dot_lkl_z_302;
                        double dot_lkl_y_310 = trr_10y * dot_lkl_z_300;
                        double dot_lkl_y_311 = trr_10y * dot_lkl_z_301;
                        double dot_lkl_y_320 = trr_20y * dot_lkl_z_300;
                        vj_ij_001 += dot_lkl_y_001 + trr_01x * dot_lkl_y_101 + trr_02x * dot_lkl_y_201 + trr_03x * dot_lkl_y_301;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202 + trr_03x * dot_lkl_y_302;
                        vj_ij_010 += dot_lkl_y_010 + trr_01x * dot_lkl_y_110 + trr_02x * dot_lkl_y_210 + trr_03x * dot_lkl_y_310;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211 + trr_03x * dot_lkl_y_311;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220 + trr_03x * dot_lkl_y_320;
                        vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100 + trr_12x * dot_lkl_y_200 + trr_13x * dot_lkl_y_300;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201 + trr_13x * dot_lkl_y_301;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210 + trr_13x * dot_lkl_y_310;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200 + trr_23x * dot_lkl_y_300;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_002 += __shfl_down_sync(mask, vj_kl_002, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+192] += vj_kl_002;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_003 += __shfl_down_sync(mask, vj_kl_003, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+288] += vj_kl_003;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_011 += __shfl_down_sync(mask, vj_kl_011, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+480] += vj_kl_011;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_012 += __shfl_down_sync(mask, vj_kl_012, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+576] += vj_kl_012;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_020 += __shfl_down_sync(mask, vj_kl_020, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+672] += vj_kl_020;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_021 += __shfl_down_sync(mask, vj_kl_021, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+768] += vj_kl_021;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_030 += __shfl_down_sync(mask, vj_kl_030, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+864] += vj_kl_030;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_101 += __shfl_down_sync(mask, vj_kl_101, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1056] += vj_kl_101;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_102 += __shfl_down_sync(mask, vj_kl_102, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1152] += vj_kl_102;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_110 += __shfl_down_sync(mask, vj_kl_110, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1248] += vj_kl_110;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_111 += __shfl_down_sync(mask, vj_kl_111, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1344] += vj_kl_111;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_120 += __shfl_down_sync(mask, vj_kl_120, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1440] += vj_kl_120;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_200 += __shfl_down_sync(mask, vj_kl_200, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1536] += vj_kl_200;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_201 += __shfl_down_sync(mask, vj_kl_201, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1632] += vj_kl_201;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_210 += __shfl_down_sync(mask, vj_kl_210, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1728] += vj_kl_210;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_300 += __shfl_down_sync(mask, vj_kl_300, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1824] += vj_kl_300;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_001;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+1, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_010;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+4, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_100;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 120; n += 16) {
        int i = n / 6;
        int tile = n % 6;
        int task_kl = blockIdx.y * 96 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*96]);
        }
    }
}

// TILEX=32, TILEY=32,
__global__
void rys_j_3_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 512;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 512;
    double *dm_ij_cache = dm_kl_cache + 512;
    double *rjri = dm_ij_cache + 320;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 512; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_030 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_120 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_210 = 0;
        double vj_ij_300 = 0;
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*32+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_000 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(2, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double dot_lij_z_000 = trr_20z * dm_ij_cache[tx+32] + trr_30z * dm_ij_cache[tx+48];
                        double dot_lij_z_010 = trr_10z * dm_ij_cache[tx+80] + trr_20z * dm_ij_cache[tx+96];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+112] + trr_10z * dm_ij_cache[tx+128];
                        double dot_lij_z_030 = wt * dm_ij_cache[tx+144];
                        double dot_lij_z_100 = trr_10z * dm_ij_cache[tx+176] + trr_20z * dm_ij_cache[tx+192];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+208] + trr_10z * dm_ij_cache[tx+224];
                        double dot_lij_z_120 = wt * dm_ij_cache[tx+240];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+256] + trr_10z * dm_ij_cache[tx+272];
                        double dot_lij_z_210 = wt * dm_ij_cache[tx+288];
                        double dot_lij_z_300 = wt * dm_ij_cache[tx+304];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120;
                        double dot_lij_y_200 = dot_lij_z_200 + trr_10y * dot_lij_z_210;
                        double dot_lij_y_300 = dot_lij_z_300;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        vj_kl_000 += dot_lij_y_000 + trr_10x * dot_lij_y_100 + trr_20x * dot_lij_y_200 + trr_30x * dot_lij_y_300;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = wt * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_001 = trr_10z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_002 = trr_20z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_003 = trr_30z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_y_000 = dot_lkl_z_000;
                        double dot_lkl_y_001 = dot_lkl_z_001;
                        double dot_lkl_y_002 = dot_lkl_z_002;
                        double dot_lkl_y_003 = dot_lkl_z_003;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001;
                        double dot_lkl_y_012 = trr_10y * dot_lkl_z_002;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000;
                        double dot_lkl_y_021 = trr_20y * dot_lkl_z_001;
                        double dot_lkl_y_030 = trr_30y * dot_lkl_z_000;
                        vj_ij_002 += dot_lkl_y_002;
                        vj_ij_003 += dot_lkl_y_003;
                        vj_ij_011 += dot_lkl_y_011;
                        vj_ij_012 += dot_lkl_y_012;
                        vj_ij_020 += dot_lkl_y_020;
                        vj_ij_021 += dot_lkl_y_021;
                        vj_ij_030 += dot_lkl_y_030;
                        vj_ij_101 += trr_10x * dot_lkl_y_001;
                        vj_ij_102 += trr_10x * dot_lkl_y_002;
                        vj_ij_110 += trr_10x * dot_lkl_y_010;
                        vj_ij_111 += trr_10x * dot_lkl_y_011;
                        vj_ij_120 += trr_10x * dot_lkl_y_020;
                        vj_ij_200 += trr_20x * dot_lkl_y_000;
                        vj_ij_201 += trr_20x * dot_lkl_y_001;
                        vj_ij_210 += trr_20x * dot_lkl_y_010;
                        vj_ij_300 += trr_30x * dot_lkl_y_000;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_000 += __shfl_down_sync(mask, vj_kl_000, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+0] += vj_kl_000;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_003;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_012;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_021;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_030;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+11, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_102;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+12, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+13, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_111;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+14, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_120;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+15, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+16, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_201;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+17, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_210;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+18, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_300;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+19, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=32, TILEY=29,
__global__
void rys_j_3_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 464;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 1856;
    double *dm_ij_cache = dm_kl_cache + 1856;
    double *rjri = dm_ij_cache + 320;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 116; n += 16) {
        int i = n / 29;
        int tile = n % 29;
        int task_kl = blockIdx.y * 464 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*464] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 1856; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_030 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_120 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_210 = 0;
        double vj_ij_300 = 0;
        for (int batch_kl = 0; batch_kl < 29; ++batch_kl) {
            int task_kl0 = blockIdx.y * 464 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*29+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_010 = 0;
            double vj_kl_100 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(3, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double dot_lij_z_000 = trr_20z * dm_ij_cache[tx+32] + trr_30z * dm_ij_cache[tx+48];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double dot_lij_z_001 = trr_21z * dm_ij_cache[tx+32] + trr_31z * dm_ij_cache[tx+48];
                        double dot_lij_z_010 = trr_10z * dm_ij_cache[tx+80] + trr_20z * dm_ij_cache[tx+96];
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double dot_lij_z_011 = trr_11z * dm_ij_cache[tx+80] + trr_21z * dm_ij_cache[tx+96];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+112] + trr_10z * dm_ij_cache[tx+128];
                        double trr_01z = cpz * wt;
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+112] + trr_11z * dm_ij_cache[tx+128];
                        double dot_lij_z_030 = wt * dm_ij_cache[tx+144];
                        double dot_lij_z_031 = trr_01z * dm_ij_cache[tx+144];
                        double dot_lij_z_100 = trr_10z * dm_ij_cache[tx+176] + trr_20z * dm_ij_cache[tx+192];
                        double dot_lij_z_101 = trr_11z * dm_ij_cache[tx+176] + trr_21z * dm_ij_cache[tx+192];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+208] + trr_10z * dm_ij_cache[tx+224];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+208] + trr_11z * dm_ij_cache[tx+224];
                        double dot_lij_z_120 = wt * dm_ij_cache[tx+240];
                        double dot_lij_z_121 = trr_01z * dm_ij_cache[tx+240];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+256] + trr_10z * dm_ij_cache[tx+272];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+256] + trr_11z * dm_ij_cache[tx+272];
                        double dot_lij_z_210 = wt * dm_ij_cache[tx+288];
                        double dot_lij_z_211 = trr_01z * dm_ij_cache[tx+288];
                        double dot_lij_z_300 = wt * dm_ij_cache[tx+304];
                        double dot_lij_z_301 = trr_01z * dm_ij_cache[tx+304];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120;
                        double dot_lij_y_200 = dot_lij_z_200 + trr_10y * dot_lij_z_210;
                        double dot_lij_y_201 = dot_lij_z_201 + trr_10y * dot_lij_z_211;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210;
                        double dot_lij_y_300 = dot_lij_z_300;
                        double dot_lij_y_301 = dot_lij_z_301;
                        double dot_lij_y_310 = trr_01y * dot_lij_z_300;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201 + trr_30x * dot_lij_y_301;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210 + trr_30x * dot_lij_y_310;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200 + trr_31x * dot_lij_y_300;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+464];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+464];
                        double dot_lkl_z_002 = trr_21z * dm_kl_cache[sq_kl+464];
                        double dot_lkl_z_003 = trr_31z * dm_kl_cache[sq_kl+464];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+928];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+928];
                        double dot_lkl_z_012 = trr_20z * dm_kl_cache[sq_kl+928];
                        double dot_lkl_z_013 = trr_30z * dm_kl_cache[sq_kl+928];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+1392];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+1392];
                        double dot_lkl_z_102 = trr_20z * dm_kl_cache[sq_kl+1392];
                        double dot_lkl_z_103 = trr_30z * dm_kl_cache[sq_kl+1392];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012;
                        double dot_lkl_y_003 = dot_lkl_z_003 + trr_01y * dot_lkl_z_013;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011;
                        double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010;
                        double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011;
                        double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010;
                        double dot_lkl_y_100 = dot_lkl_z_100;
                        double dot_lkl_y_101 = dot_lkl_z_101;
                        double dot_lkl_y_102 = dot_lkl_z_102;
                        double dot_lkl_y_103 = dot_lkl_z_103;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101;
                        double dot_lkl_y_112 = trr_10y * dot_lkl_z_102;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100;
                        double dot_lkl_y_121 = trr_20y * dot_lkl_z_101;
                        double dot_lkl_y_130 = trr_30y * dot_lkl_z_100;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102;
                        vj_ij_003 += dot_lkl_y_003 + trr_01x * dot_lkl_y_103;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111;
                        vj_ij_012 += dot_lkl_y_012 + trr_01x * dot_lkl_y_112;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120;
                        vj_ij_021 += dot_lkl_y_021 + trr_01x * dot_lkl_y_121;
                        vj_ij_030 += dot_lkl_y_030 + trr_01x * dot_lkl_y_130;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101;
                        vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110;
                        vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111;
                        vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100;
                        vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101;
                        vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110;
                        vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+464] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+928] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1392] += vj_kl_100;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_003;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_012;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_021;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_030;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+11, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_102;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+12, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+13, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_111;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+14, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_120;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+15, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+16, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_201;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+17, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_210;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+18, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_300;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+19, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 116; n += 16) {
        int i = n / 29;
        int tile = n % 29;
        int task_kl = blockIdx.y * 464 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*464]);
        }
    }
}

// TILEX=32, TILEY=11,
__global__
void rys_j_3_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 176;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 1760;
    double *dm_ij_cache = dm_kl_cache + 1760;
    double *rjri = dm_ij_cache + 320;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 110; n += 16) {
        int i = n / 11;
        int tile = n % 11;
        int task_kl = blockIdx.y * 176 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*176] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 1760; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_030 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_120 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_210 = 0;
        double vj_ij_300 = 0;
        for (int batch_kl = 0; batch_kl < 11; ++batch_kl) {
            int task_kl0 = blockIdx.y * 176 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*11+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_002 = 0;
            double vj_kl_010 = 0;
            double vj_kl_011 = 0;
            double vj_kl_020 = 0;
            double vj_kl_100 = 0;
            double vj_kl_101 = 0;
            double vj_kl_110 = 0;
            double vj_kl_200 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(3, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double dot_lij_z_000 = trr_20z * dm_ij_cache[tx+32] + trr_30z * dm_ij_cache[tx+48];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double b01 = .5/akl_cache[ty] * (1 - rt_akl);
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double dot_lij_z_001 = trr_21z * dm_ij_cache[tx+32] + trr_31z * dm_ij_cache[tx+48];
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                        double dot_lij_z_002 = trr_22z * dm_ij_cache[tx+32] + trr_32z * dm_ij_cache[tx+48];
                        double dot_lij_z_010 = trr_10z * dm_ij_cache[tx+80] + trr_20z * dm_ij_cache[tx+96];
                        double dot_lij_z_011 = trr_11z * dm_ij_cache[tx+80] + trr_21z * dm_ij_cache[tx+96];
                        double trr_01z = cpz * wt;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double dot_lij_z_012 = trr_12z * dm_ij_cache[tx+80] + trr_22z * dm_ij_cache[tx+96];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+112] + trr_10z * dm_ij_cache[tx+128];
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+112] + trr_11z * dm_ij_cache[tx+128];
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double dot_lij_z_022 = trr_02z * dm_ij_cache[tx+112] + trr_12z * dm_ij_cache[tx+128];
                        double dot_lij_z_030 = wt * dm_ij_cache[tx+144];
                        double dot_lij_z_031 = trr_01z * dm_ij_cache[tx+144];
                        double dot_lij_z_032 = trr_02z * dm_ij_cache[tx+144];
                        double dot_lij_z_100 = trr_10z * dm_ij_cache[tx+176] + trr_20z * dm_ij_cache[tx+192];
                        double dot_lij_z_101 = trr_11z * dm_ij_cache[tx+176] + trr_21z * dm_ij_cache[tx+192];
                        double dot_lij_z_102 = trr_12z * dm_ij_cache[tx+176] + trr_22z * dm_ij_cache[tx+192];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+208] + trr_10z * dm_ij_cache[tx+224];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+208] + trr_11z * dm_ij_cache[tx+224];
                        double dot_lij_z_112 = trr_02z * dm_ij_cache[tx+208] + trr_12z * dm_ij_cache[tx+224];
                        double dot_lij_z_120 = wt * dm_ij_cache[tx+240];
                        double dot_lij_z_121 = trr_01z * dm_ij_cache[tx+240];
                        double dot_lij_z_122 = trr_02z * dm_ij_cache[tx+240];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+256] + trr_10z * dm_ij_cache[tx+272];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+256] + trr_11z * dm_ij_cache[tx+272];
                        double dot_lij_z_202 = trr_02z * dm_ij_cache[tx+256] + trr_12z * dm_ij_cache[tx+272];
                        double dot_lij_z_210 = wt * dm_ij_cache[tx+288];
                        double dot_lij_z_211 = trr_01z * dm_ij_cache[tx+288];
                        double dot_lij_z_212 = trr_02z * dm_ij_cache[tx+288];
                        double dot_lij_z_300 = wt * dm_ij_cache[tx+304];
                        double dot_lij_z_301 = trr_01z * dm_ij_cache[tx+304];
                        double dot_lij_z_302 = trr_02z * dm_ij_cache[tx+304];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031;
                        double dot_lij_y_002 = dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022 + trr_30y * dot_lij_z_032;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030;
                        double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021 + trr_31y * dot_lij_z_031;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                        double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020 + trr_32y * dot_lij_z_030;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121;
                        double dot_lij_y_102 = dot_lij_z_102 + trr_10y * dot_lij_z_112 + trr_20y * dot_lij_z_122;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120;
                        double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111 + trr_21y * dot_lij_z_121;
                        double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110 + trr_22y * dot_lij_z_120;
                        double dot_lij_y_200 = dot_lij_z_200 + trr_10y * dot_lij_z_210;
                        double dot_lij_y_201 = dot_lij_z_201 + trr_10y * dot_lij_z_211;
                        double dot_lij_y_202 = dot_lij_z_202 + trr_10y * dot_lij_z_212;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210;
                        double dot_lij_y_211 = trr_01y * dot_lij_z_201 + trr_11y * dot_lij_z_211;
                        double dot_lij_y_220 = trr_02y * dot_lij_z_200 + trr_12y * dot_lij_z_210;
                        double dot_lij_y_300 = dot_lij_z_300;
                        double dot_lij_y_301 = dot_lij_z_301;
                        double dot_lij_y_302 = dot_lij_z_302;
                        double dot_lij_y_310 = trr_01y * dot_lij_z_300;
                        double dot_lij_y_311 = trr_01y * dot_lij_z_301;
                        double dot_lij_y_320 = trr_02y * dot_lij_z_300;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201 + trr_30x * dot_lij_y_301;
                        vj_kl_002 += dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202 + trr_30x * dot_lij_y_302;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210 + trr_30x * dot_lij_y_310;
                        vj_kl_011 += dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211 + trr_30x * dot_lij_y_311;
                        vj_kl_020 += dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220 + trr_30x * dot_lij_y_320;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200 + trr_31x * dot_lij_y_300;
                        vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201 + trr_31x * dot_lij_y_301;
                        vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210 + trr_31x * dot_lij_y_310;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                        vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200 + trr_32x * dot_lij_y_300;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+176] + trr_02z * dm_kl_cache[sq_kl+352];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+176] + trr_12z * dm_kl_cache[sq_kl+352];
                        double dot_lkl_z_002 = trr_21z * dm_kl_cache[sq_kl+176] + trr_22z * dm_kl_cache[sq_kl+352];
                        double dot_lkl_z_003 = trr_31z * dm_kl_cache[sq_kl+176] + trr_32z * dm_kl_cache[sq_kl+352];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+528] + trr_01z * dm_kl_cache[sq_kl+704];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+528] + trr_11z * dm_kl_cache[sq_kl+704];
                        double dot_lkl_z_012 = trr_20z * dm_kl_cache[sq_kl+528] + trr_21z * dm_kl_cache[sq_kl+704];
                        double dot_lkl_z_013 = trr_30z * dm_kl_cache[sq_kl+528] + trr_31z * dm_kl_cache[sq_kl+704];
                        double dot_lkl_z_020 = wt * dm_kl_cache[sq_kl+880];
                        double dot_lkl_z_021 = trr_10z * dm_kl_cache[sq_kl+880];
                        double dot_lkl_z_022 = trr_20z * dm_kl_cache[sq_kl+880];
                        double dot_lkl_z_023 = trr_30z * dm_kl_cache[sq_kl+880];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+1056] + trr_01z * dm_kl_cache[sq_kl+1232];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+1056] + trr_11z * dm_kl_cache[sq_kl+1232];
                        double dot_lkl_z_102 = trr_20z * dm_kl_cache[sq_kl+1056] + trr_21z * dm_kl_cache[sq_kl+1232];
                        double dot_lkl_z_103 = trr_30z * dm_kl_cache[sq_kl+1056] + trr_31z * dm_kl_cache[sq_kl+1232];
                        double dot_lkl_z_110 = wt * dm_kl_cache[sq_kl+1408];
                        double dot_lkl_z_111 = trr_10z * dm_kl_cache[sq_kl+1408];
                        double dot_lkl_z_112 = trr_20z * dm_kl_cache[sq_kl+1408];
                        double dot_lkl_z_113 = trr_30z * dm_kl_cache[sq_kl+1408];
                        double dot_lkl_z_200 = wt * dm_kl_cache[sq_kl+1584];
                        double dot_lkl_z_201 = trr_10z * dm_kl_cache[sq_kl+1584];
                        double dot_lkl_z_202 = trr_20z * dm_kl_cache[sq_kl+1584];
                        double dot_lkl_z_203 = trr_30z * dm_kl_cache[sq_kl+1584];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022;
                        double dot_lkl_y_003 = dot_lkl_z_003 + trr_01y * dot_lkl_z_013 + trr_02y * dot_lkl_z_023;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021;
                        double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012 + trr_12y * dot_lkl_z_022;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020;
                        double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011 + trr_22y * dot_lkl_z_021;
                        double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010 + trr_32y * dot_lkl_z_020;
                        double dot_lkl_y_100 = dot_lkl_z_100 + trr_01y * dot_lkl_z_110;
                        double dot_lkl_y_101 = dot_lkl_z_101 + trr_01y * dot_lkl_z_111;
                        double dot_lkl_y_102 = dot_lkl_z_102 + trr_01y * dot_lkl_z_112;
                        double dot_lkl_y_103 = dot_lkl_z_103 + trr_01y * dot_lkl_z_113;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111;
                        double dot_lkl_y_112 = trr_10y * dot_lkl_z_102 + trr_11y * dot_lkl_z_112;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110;
                        double dot_lkl_y_121 = trr_20y * dot_lkl_z_101 + trr_21y * dot_lkl_z_111;
                        double dot_lkl_y_130 = trr_30y * dot_lkl_z_100 + trr_31y * dot_lkl_z_110;
                        double dot_lkl_y_200 = dot_lkl_z_200;
                        double dot_lkl_y_201 = dot_lkl_z_201;
                        double dot_lkl_y_202 = dot_lkl_z_202;
                        double dot_lkl_y_203 = dot_lkl_z_203;
                        double dot_lkl_y_210 = trr_10y * dot_lkl_z_200;
                        double dot_lkl_y_211 = trr_10y * dot_lkl_z_201;
                        double dot_lkl_y_212 = trr_10y * dot_lkl_z_202;
                        double dot_lkl_y_220 = trr_20y * dot_lkl_z_200;
                        double dot_lkl_y_221 = trr_20y * dot_lkl_z_201;
                        double dot_lkl_y_230 = trr_30y * dot_lkl_z_200;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202;
                        vj_ij_003 += dot_lkl_y_003 + trr_01x * dot_lkl_y_103 + trr_02x * dot_lkl_y_203;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211;
                        vj_ij_012 += dot_lkl_y_012 + trr_01x * dot_lkl_y_112 + trr_02x * dot_lkl_y_212;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220;
                        vj_ij_021 += dot_lkl_y_021 + trr_01x * dot_lkl_y_121 + trr_02x * dot_lkl_y_221;
                        vj_ij_030 += dot_lkl_y_030 + trr_01x * dot_lkl_y_130 + trr_02x * dot_lkl_y_230;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201;
                        vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102 + trr_12x * dot_lkl_y_202;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210;
                        vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111 + trr_12x * dot_lkl_y_211;
                        vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120 + trr_12x * dot_lkl_y_220;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200;
                        vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101 + trr_22x * dot_lkl_y_201;
                        vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110 + trr_22x * dot_lkl_y_210;
                        vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100 + trr_32x * dot_lkl_y_200;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+176] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_002 += __shfl_down_sync(mask, vj_kl_002, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+352] += vj_kl_002;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+528] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_011 += __shfl_down_sync(mask, vj_kl_011, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+704] += vj_kl_011;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_020 += __shfl_down_sync(mask, vj_kl_020, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+880] += vj_kl_020;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1056] += vj_kl_100;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_101 += __shfl_down_sync(mask, vj_kl_101, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1232] += vj_kl_101;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_110 += __shfl_down_sync(mask, vj_kl_110, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1408] += vj_kl_110;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_200 += __shfl_down_sync(mask, vj_kl_200, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1584] += vj_kl_200;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_003;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_012;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_021;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_030;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+11, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_102;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+12, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+13, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_111;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+14, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_120;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+15, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+16, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_201;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+17, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_210;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+18, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_300;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+19, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 110; n += 16) {
        int i = n / 11;
        int tile = n % 11;
        int task_kl = blockIdx.y * 176 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*176]);
        }
    }
}

// TILEX=32, TILEY=5,
__global__
void rys_j_3_3(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 80;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 1600;
    double *dm_ij_cache = dm_kl_cache + 1600;
    double *rjri = dm_ij_cache + 320;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 100; n += 16) {
        int i = n / 5;
        int tile = n % 5;
        int task_kl = blockIdx.y * 80 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*80] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 1600; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_030 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_120 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_210 = 0;
        double vj_ij_300 = 0;
        for (int batch_kl = 0; batch_kl < 5; ++batch_kl) {
            int task_kl0 = blockIdx.y * 80 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*5+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_002 = 0;
            double vj_kl_003 = 0;
            double vj_kl_011 = 0;
            double vj_kl_012 = 0;
            double vj_kl_020 = 0;
            double vj_kl_021 = 0;
            double vj_kl_030 = 0;
            double vj_kl_101 = 0;
            double vj_kl_102 = 0;
            double vj_kl_110 = 0;
            double vj_kl_111 = 0;
            double vj_kl_120 = 0;
            double vj_kl_200 = 0;
            double vj_kl_201 = 0;
            double vj_kl_210 = 0;
            double vj_kl_300 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(4, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 4; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double dot_lij_z_000 = trr_20z * dm_ij_cache[tx+32] + trr_30z * dm_ij_cache[tx+48];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double b01 = .5/akl_cache[ty] * (1 - rt_akl);
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double dot_lij_z_001 = trr_21z * dm_ij_cache[tx+32] + trr_31z * dm_ij_cache[tx+48];
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                        double dot_lij_z_002 = trr_22z * dm_ij_cache[tx+32] + trr_32z * dm_ij_cache[tx+48];
                        double trr_01z = cpz * wt;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                        double trr_33z = cpz * trr_32z + 2*b01 * trr_31z + 3*b00 * trr_22z;
                        double dot_lij_z_003 = trr_23z * dm_ij_cache[tx+32] + trr_33z * dm_ij_cache[tx+48];
                        double dot_lij_z_010 = trr_10z * dm_ij_cache[tx+80] + trr_20z * dm_ij_cache[tx+96];
                        double dot_lij_z_011 = trr_11z * dm_ij_cache[tx+80] + trr_21z * dm_ij_cache[tx+96];
                        double dot_lij_z_012 = trr_12z * dm_ij_cache[tx+80] + trr_22z * dm_ij_cache[tx+96];
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        double dot_lij_z_013 = trr_13z * dm_ij_cache[tx+80] + trr_23z * dm_ij_cache[tx+96];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+112] + trr_10z * dm_ij_cache[tx+128];
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+112] + trr_11z * dm_ij_cache[tx+128];
                        double dot_lij_z_022 = trr_02z * dm_ij_cache[tx+112] + trr_12z * dm_ij_cache[tx+128];
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        double dot_lij_z_023 = trr_03z * dm_ij_cache[tx+112] + trr_13z * dm_ij_cache[tx+128];
                        double dot_lij_z_030 = wt * dm_ij_cache[tx+144];
                        double dot_lij_z_031 = trr_01z * dm_ij_cache[tx+144];
                        double dot_lij_z_032 = trr_02z * dm_ij_cache[tx+144];
                        double dot_lij_z_033 = trr_03z * dm_ij_cache[tx+144];
                        double dot_lij_z_100 = trr_10z * dm_ij_cache[tx+176] + trr_20z * dm_ij_cache[tx+192];
                        double dot_lij_z_101 = trr_11z * dm_ij_cache[tx+176] + trr_21z * dm_ij_cache[tx+192];
                        double dot_lij_z_102 = trr_12z * dm_ij_cache[tx+176] + trr_22z * dm_ij_cache[tx+192];
                        double dot_lij_z_103 = trr_13z * dm_ij_cache[tx+176] + trr_23z * dm_ij_cache[tx+192];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+208] + trr_10z * dm_ij_cache[tx+224];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+208] + trr_11z * dm_ij_cache[tx+224];
                        double dot_lij_z_112 = trr_02z * dm_ij_cache[tx+208] + trr_12z * dm_ij_cache[tx+224];
                        double dot_lij_z_113 = trr_03z * dm_ij_cache[tx+208] + trr_13z * dm_ij_cache[tx+224];
                        double dot_lij_z_120 = wt * dm_ij_cache[tx+240];
                        double dot_lij_z_121 = trr_01z * dm_ij_cache[tx+240];
                        double dot_lij_z_122 = trr_02z * dm_ij_cache[tx+240];
                        double dot_lij_z_123 = trr_03z * dm_ij_cache[tx+240];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+256] + trr_10z * dm_ij_cache[tx+272];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+256] + trr_11z * dm_ij_cache[tx+272];
                        double dot_lij_z_202 = trr_02z * dm_ij_cache[tx+256] + trr_12z * dm_ij_cache[tx+272];
                        double dot_lij_z_203 = trr_03z * dm_ij_cache[tx+256] + trr_13z * dm_ij_cache[tx+272];
                        double dot_lij_z_210 = wt * dm_ij_cache[tx+288];
                        double dot_lij_z_211 = trr_01z * dm_ij_cache[tx+288];
                        double dot_lij_z_212 = trr_02z * dm_ij_cache[tx+288];
                        double dot_lij_z_213 = trr_03z * dm_ij_cache[tx+288];
                        double dot_lij_z_300 = wt * dm_ij_cache[tx+304];
                        double dot_lij_z_301 = trr_01z * dm_ij_cache[tx+304];
                        double dot_lij_z_302 = trr_02z * dm_ij_cache[tx+304];
                        double dot_lij_z_303 = trr_03z * dm_ij_cache[tx+304];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031;
                        double dot_lij_y_002 = dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022 + trr_30y * dot_lij_z_032;
                        double dot_lij_y_003 = dot_lij_z_003 + trr_10y * dot_lij_z_013 + trr_20y * dot_lij_z_023 + trr_30y * dot_lij_z_033;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030;
                        double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021 + trr_31y * dot_lij_z_031;
                        double dot_lij_y_012 = trr_01y * dot_lij_z_002 + trr_11y * dot_lij_z_012 + trr_21y * dot_lij_z_022 + trr_31y * dot_lij_z_032;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                        double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020 + trr_32y * dot_lij_z_030;
                        double dot_lij_y_021 = trr_02y * dot_lij_z_001 + trr_12y * dot_lij_z_011 + trr_22y * dot_lij_z_021 + trr_32y * dot_lij_z_031;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                        double trr_33y = cpy * trr_32y + 2*b01 * trr_31y + 3*b00 * trr_22y;
                        double dot_lij_y_030 = trr_03y * dot_lij_z_000 + trr_13y * dot_lij_z_010 + trr_23y * dot_lij_z_020 + trr_33y * dot_lij_z_030;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121;
                        double dot_lij_y_102 = dot_lij_z_102 + trr_10y * dot_lij_z_112 + trr_20y * dot_lij_z_122;
                        double dot_lij_y_103 = dot_lij_z_103 + trr_10y * dot_lij_z_113 + trr_20y * dot_lij_z_123;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120;
                        double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111 + trr_21y * dot_lij_z_121;
                        double dot_lij_y_112 = trr_01y * dot_lij_z_102 + trr_11y * dot_lij_z_112 + trr_21y * dot_lij_z_122;
                        double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110 + trr_22y * dot_lij_z_120;
                        double dot_lij_y_121 = trr_02y * dot_lij_z_101 + trr_12y * dot_lij_z_111 + trr_22y * dot_lij_z_121;
                        double dot_lij_y_130 = trr_03y * dot_lij_z_100 + trr_13y * dot_lij_z_110 + trr_23y * dot_lij_z_120;
                        double dot_lij_y_200 = dot_lij_z_200 + trr_10y * dot_lij_z_210;
                        double dot_lij_y_201 = dot_lij_z_201 + trr_10y * dot_lij_z_211;
                        double dot_lij_y_202 = dot_lij_z_202 + trr_10y * dot_lij_z_212;
                        double dot_lij_y_203 = dot_lij_z_203 + trr_10y * dot_lij_z_213;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210;
                        double dot_lij_y_211 = trr_01y * dot_lij_z_201 + trr_11y * dot_lij_z_211;
                        double dot_lij_y_212 = trr_01y * dot_lij_z_202 + trr_11y * dot_lij_z_212;
                        double dot_lij_y_220 = trr_02y * dot_lij_z_200 + trr_12y * dot_lij_z_210;
                        double dot_lij_y_221 = trr_02y * dot_lij_z_201 + trr_12y * dot_lij_z_211;
                        double dot_lij_y_230 = trr_03y * dot_lij_z_200 + trr_13y * dot_lij_z_210;
                        double dot_lij_y_300 = dot_lij_z_300;
                        double dot_lij_y_301 = dot_lij_z_301;
                        double dot_lij_y_302 = dot_lij_z_302;
                        double dot_lij_y_303 = dot_lij_z_303;
                        double dot_lij_y_310 = trr_01y * dot_lij_z_300;
                        double dot_lij_y_311 = trr_01y * dot_lij_z_301;
                        double dot_lij_y_312 = trr_01y * dot_lij_z_302;
                        double dot_lij_y_320 = trr_02y * dot_lij_z_300;
                        double dot_lij_y_321 = trr_02y * dot_lij_z_301;
                        double dot_lij_y_330 = trr_03y * dot_lij_z_300;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        vj_kl_002 += dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202 + trr_30x * dot_lij_y_302;
                        vj_kl_003 += dot_lij_y_003 + trr_10x * dot_lij_y_103 + trr_20x * dot_lij_y_203 + trr_30x * dot_lij_y_303;
                        vj_kl_011 += dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211 + trr_30x * dot_lij_y_311;
                        vj_kl_012 += dot_lij_y_012 + trr_10x * dot_lij_y_112 + trr_20x * dot_lij_y_212 + trr_30x * dot_lij_y_312;
                        vj_kl_020 += dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220 + trr_30x * dot_lij_y_320;
                        vj_kl_021 += dot_lij_y_021 + trr_10x * dot_lij_y_121 + trr_20x * dot_lij_y_221 + trr_30x * dot_lij_y_321;
                        vj_kl_030 += dot_lij_y_030 + trr_10x * dot_lij_y_130 + trr_20x * dot_lij_y_230 + trr_30x * dot_lij_y_330;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201 + trr_31x * dot_lij_y_301;
                        vj_kl_102 += trr_01x * dot_lij_y_002 + trr_11x * dot_lij_y_102 + trr_21x * dot_lij_y_202 + trr_31x * dot_lij_y_302;
                        vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210 + trr_31x * dot_lij_y_310;
                        vj_kl_111 += trr_01x * dot_lij_y_011 + trr_11x * dot_lij_y_111 + trr_21x * dot_lij_y_211 + trr_31x * dot_lij_y_311;
                        vj_kl_120 += trr_01x * dot_lij_y_020 + trr_11x * dot_lij_y_120 + trr_21x * dot_lij_y_220 + trr_31x * dot_lij_y_320;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                        vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200 + trr_32x * dot_lij_y_300;
                        vj_kl_201 += trr_02x * dot_lij_y_001 + trr_12x * dot_lij_y_101 + trr_22x * dot_lij_y_201 + trr_32x * dot_lij_y_301;
                        vj_kl_210 += trr_02x * dot_lij_y_010 + trr_12x * dot_lij_y_110 + trr_22x * dot_lij_y_210 + trr_32x * dot_lij_y_310;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                        double trr_33x = cpx * trr_32x + 2*b01 * trr_31x + 3*b00 * trr_22x;
                        vj_kl_300 += trr_03x * dot_lij_y_000 + trr_13x * dot_lij_y_100 + trr_23x * dot_lij_y_200 + trr_33x * dot_lij_y_300;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_02z * dm_kl_cache[sq_kl+160] + trr_03z * dm_kl_cache[sq_kl+240];
                        double dot_lkl_z_001 = trr_12z * dm_kl_cache[sq_kl+160] + trr_13z * dm_kl_cache[sq_kl+240];
                        double dot_lkl_z_002 = trr_22z * dm_kl_cache[sq_kl+160] + trr_23z * dm_kl_cache[sq_kl+240];
                        double dot_lkl_z_003 = trr_32z * dm_kl_cache[sq_kl+160] + trr_33z * dm_kl_cache[sq_kl+240];
                        double dot_lkl_z_010 = trr_01z * dm_kl_cache[sq_kl+400] + trr_02z * dm_kl_cache[sq_kl+480];
                        double dot_lkl_z_011 = trr_11z * dm_kl_cache[sq_kl+400] + trr_12z * dm_kl_cache[sq_kl+480];
                        double dot_lkl_z_012 = trr_21z * dm_kl_cache[sq_kl+400] + trr_22z * dm_kl_cache[sq_kl+480];
                        double dot_lkl_z_013 = trr_31z * dm_kl_cache[sq_kl+400] + trr_32z * dm_kl_cache[sq_kl+480];
                        double dot_lkl_z_020 = wt * dm_kl_cache[sq_kl+560] + trr_01z * dm_kl_cache[sq_kl+640];
                        double dot_lkl_z_021 = trr_10z * dm_kl_cache[sq_kl+560] + trr_11z * dm_kl_cache[sq_kl+640];
                        double dot_lkl_z_022 = trr_20z * dm_kl_cache[sq_kl+560] + trr_21z * dm_kl_cache[sq_kl+640];
                        double dot_lkl_z_023 = trr_30z * dm_kl_cache[sq_kl+560] + trr_31z * dm_kl_cache[sq_kl+640];
                        double dot_lkl_z_030 = wt * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_031 = trr_10z * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_032 = trr_20z * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_033 = trr_30z * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_100 = trr_01z * dm_kl_cache[sq_kl+880] + trr_02z * dm_kl_cache[sq_kl+960];
                        double dot_lkl_z_101 = trr_11z * dm_kl_cache[sq_kl+880] + trr_12z * dm_kl_cache[sq_kl+960];
                        double dot_lkl_z_102 = trr_21z * dm_kl_cache[sq_kl+880] + trr_22z * dm_kl_cache[sq_kl+960];
                        double dot_lkl_z_103 = trr_31z * dm_kl_cache[sq_kl+880] + trr_32z * dm_kl_cache[sq_kl+960];
                        double dot_lkl_z_110 = wt * dm_kl_cache[sq_kl+1040] + trr_01z * dm_kl_cache[sq_kl+1120];
                        double dot_lkl_z_111 = trr_10z * dm_kl_cache[sq_kl+1040] + trr_11z * dm_kl_cache[sq_kl+1120];
                        double dot_lkl_z_112 = trr_20z * dm_kl_cache[sq_kl+1040] + trr_21z * dm_kl_cache[sq_kl+1120];
                        double dot_lkl_z_113 = trr_30z * dm_kl_cache[sq_kl+1040] + trr_31z * dm_kl_cache[sq_kl+1120];
                        double dot_lkl_z_120 = wt * dm_kl_cache[sq_kl+1200];
                        double dot_lkl_z_121 = trr_10z * dm_kl_cache[sq_kl+1200];
                        double dot_lkl_z_122 = trr_20z * dm_kl_cache[sq_kl+1200];
                        double dot_lkl_z_123 = trr_30z * dm_kl_cache[sq_kl+1200];
                        double dot_lkl_z_200 = wt * dm_kl_cache[sq_kl+1280] + trr_01z * dm_kl_cache[sq_kl+1360];
                        double dot_lkl_z_201 = trr_10z * dm_kl_cache[sq_kl+1280] + trr_11z * dm_kl_cache[sq_kl+1360];
                        double dot_lkl_z_202 = trr_20z * dm_kl_cache[sq_kl+1280] + trr_21z * dm_kl_cache[sq_kl+1360];
                        double dot_lkl_z_203 = trr_30z * dm_kl_cache[sq_kl+1280] + trr_31z * dm_kl_cache[sq_kl+1360];
                        double dot_lkl_z_210 = wt * dm_kl_cache[sq_kl+1440];
                        double dot_lkl_z_211 = trr_10z * dm_kl_cache[sq_kl+1440];
                        double dot_lkl_z_212 = trr_20z * dm_kl_cache[sq_kl+1440];
                        double dot_lkl_z_213 = trr_30z * dm_kl_cache[sq_kl+1440];
                        double dot_lkl_z_300 = wt * dm_kl_cache[sq_kl+1520];
                        double dot_lkl_z_301 = trr_10z * dm_kl_cache[sq_kl+1520];
                        double dot_lkl_z_302 = trr_20z * dm_kl_cache[sq_kl+1520];
                        double dot_lkl_z_303 = trr_30z * dm_kl_cache[sq_kl+1520];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020 + trr_03y * dot_lkl_z_030;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021 + trr_03y * dot_lkl_z_031;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022 + trr_03y * dot_lkl_z_032;
                        double dot_lkl_y_003 = dot_lkl_z_003 + trr_01y * dot_lkl_z_013 + trr_02y * dot_lkl_z_023 + trr_03y * dot_lkl_z_033;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020 + trr_13y * dot_lkl_z_030;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021 + trr_13y * dot_lkl_z_031;
                        double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012 + trr_12y * dot_lkl_z_022 + trr_13y * dot_lkl_z_032;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020 + trr_23y * dot_lkl_z_030;
                        double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011 + trr_22y * dot_lkl_z_021 + trr_23y * dot_lkl_z_031;
                        double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010 + trr_32y * dot_lkl_z_020 + trr_33y * dot_lkl_z_030;
                        double dot_lkl_y_100 = dot_lkl_z_100 + trr_01y * dot_lkl_z_110 + trr_02y * dot_lkl_z_120;
                        double dot_lkl_y_101 = dot_lkl_z_101 + trr_01y * dot_lkl_z_111 + trr_02y * dot_lkl_z_121;
                        double dot_lkl_y_102 = dot_lkl_z_102 + trr_01y * dot_lkl_z_112 + trr_02y * dot_lkl_z_122;
                        double dot_lkl_y_103 = dot_lkl_z_103 + trr_01y * dot_lkl_z_113 + trr_02y * dot_lkl_z_123;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110 + trr_12y * dot_lkl_z_120;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111 + trr_12y * dot_lkl_z_121;
                        double dot_lkl_y_112 = trr_10y * dot_lkl_z_102 + trr_11y * dot_lkl_z_112 + trr_12y * dot_lkl_z_122;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110 + trr_22y * dot_lkl_z_120;
                        double dot_lkl_y_121 = trr_20y * dot_lkl_z_101 + trr_21y * dot_lkl_z_111 + trr_22y * dot_lkl_z_121;
                        double dot_lkl_y_130 = trr_30y * dot_lkl_z_100 + trr_31y * dot_lkl_z_110 + trr_32y * dot_lkl_z_120;
                        double dot_lkl_y_200 = dot_lkl_z_200 + trr_01y * dot_lkl_z_210;
                        double dot_lkl_y_201 = dot_lkl_z_201 + trr_01y * dot_lkl_z_211;
                        double dot_lkl_y_202 = dot_lkl_z_202 + trr_01y * dot_lkl_z_212;
                        double dot_lkl_y_203 = dot_lkl_z_203 + trr_01y * dot_lkl_z_213;
                        double dot_lkl_y_210 = trr_10y * dot_lkl_z_200 + trr_11y * dot_lkl_z_210;
                        double dot_lkl_y_211 = trr_10y * dot_lkl_z_201 + trr_11y * dot_lkl_z_211;
                        double dot_lkl_y_212 = trr_10y * dot_lkl_z_202 + trr_11y * dot_lkl_z_212;
                        double dot_lkl_y_220 = trr_20y * dot_lkl_z_200 + trr_21y * dot_lkl_z_210;
                        double dot_lkl_y_221 = trr_20y * dot_lkl_z_201 + trr_21y * dot_lkl_z_211;
                        double dot_lkl_y_230 = trr_30y * dot_lkl_z_200 + trr_31y * dot_lkl_z_210;
                        double dot_lkl_y_300 = dot_lkl_z_300;
                        double dot_lkl_y_301 = dot_lkl_z_301;
                        double dot_lkl_y_302 = dot_lkl_z_302;
                        double dot_lkl_y_303 = dot_lkl_z_303;
                        double dot_lkl_y_310 = trr_10y * dot_lkl_z_300;
                        double dot_lkl_y_311 = trr_10y * dot_lkl_z_301;
                        double dot_lkl_y_312 = trr_10y * dot_lkl_z_302;
                        double dot_lkl_y_320 = trr_20y * dot_lkl_z_300;
                        double dot_lkl_y_321 = trr_20y * dot_lkl_z_301;
                        double dot_lkl_y_330 = trr_30y * dot_lkl_z_300;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202 + trr_03x * dot_lkl_y_302;
                        vj_ij_003 += dot_lkl_y_003 + trr_01x * dot_lkl_y_103 + trr_02x * dot_lkl_y_203 + trr_03x * dot_lkl_y_303;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211 + trr_03x * dot_lkl_y_311;
                        vj_ij_012 += dot_lkl_y_012 + trr_01x * dot_lkl_y_112 + trr_02x * dot_lkl_y_212 + trr_03x * dot_lkl_y_312;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220 + trr_03x * dot_lkl_y_320;
                        vj_ij_021 += dot_lkl_y_021 + trr_01x * dot_lkl_y_121 + trr_02x * dot_lkl_y_221 + trr_03x * dot_lkl_y_321;
                        vj_ij_030 += dot_lkl_y_030 + trr_01x * dot_lkl_y_130 + trr_02x * dot_lkl_y_230 + trr_03x * dot_lkl_y_330;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201 + trr_13x * dot_lkl_y_301;
                        vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102 + trr_12x * dot_lkl_y_202 + trr_13x * dot_lkl_y_302;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210 + trr_13x * dot_lkl_y_310;
                        vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111 + trr_12x * dot_lkl_y_211 + trr_13x * dot_lkl_y_311;
                        vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120 + trr_12x * dot_lkl_y_220 + trr_13x * dot_lkl_y_320;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200 + trr_23x * dot_lkl_y_300;
                        vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101 + trr_22x * dot_lkl_y_201 + trr_23x * dot_lkl_y_301;
                        vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110 + trr_22x * dot_lkl_y_210 + trr_23x * dot_lkl_y_310;
                        vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100 + trr_32x * dot_lkl_y_200 + trr_33x * dot_lkl_y_300;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_002 += __shfl_down_sync(mask, vj_kl_002, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+160] += vj_kl_002;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_003 += __shfl_down_sync(mask, vj_kl_003, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+240] += vj_kl_003;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_011 += __shfl_down_sync(mask, vj_kl_011, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+400] += vj_kl_011;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_012 += __shfl_down_sync(mask, vj_kl_012, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+480] += vj_kl_012;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_020 += __shfl_down_sync(mask, vj_kl_020, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+560] += vj_kl_020;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_021 += __shfl_down_sync(mask, vj_kl_021, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+640] += vj_kl_021;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_030 += __shfl_down_sync(mask, vj_kl_030, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+720] += vj_kl_030;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_101 += __shfl_down_sync(mask, vj_kl_101, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+880] += vj_kl_101;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_102 += __shfl_down_sync(mask, vj_kl_102, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+960] += vj_kl_102;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_110 += __shfl_down_sync(mask, vj_kl_110, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1040] += vj_kl_110;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_111 += __shfl_down_sync(mask, vj_kl_111, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1120] += vj_kl_111;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_120 += __shfl_down_sync(mask, vj_kl_120, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1200] += vj_kl_120;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_200 += __shfl_down_sync(mask, vj_kl_200, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1280] += vj_kl_200;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_201 += __shfl_down_sync(mask, vj_kl_201, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1360] += vj_kl_201;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_210 += __shfl_down_sync(mask, vj_kl_210, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1440] += vj_kl_210;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_300 += __shfl_down_sync(mask, vj_kl_300, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1520] += vj_kl_300;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_003;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+5, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_012;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_021;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_030;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+11, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_102;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+12, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+13, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_111;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+14, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_120;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+15, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+16, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_201;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+17, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_210;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+18, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_300;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+19, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 100; n += 16) {
        int i = n / 5;
        int tile = n % 5;
        int task_kl = blockIdx.y * 80 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*80]);
        }
    }
}

// TILEX=32, TILEY=32,
__global__
void rys_j_4_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 512;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 512;
    double *dm_ij_cache = dm_kl_cache + 512;
    double *rjri = dm_ij_cache + 560;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 512; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_004 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_013 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_022 = 0;
        double vj_ij_030 = 0;
        double vj_ij_031 = 0;
        double vj_ij_040 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_103 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_112 = 0;
        double vj_ij_120 = 0;
        double vj_ij_121 = 0;
        double vj_ij_130 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_202 = 0;
        double vj_ij_210 = 0;
        double vj_ij_211 = 0;
        double vj_ij_220 = 0;
        double vj_ij_300 = 0;
        double vj_ij_301 = 0;
        double vj_ij_310 = 0;
        double vj_ij_400 = 0;
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*32+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_000 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(3, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double dot_lij_z_000 = trr_20z * dm_ij_cache[tx+32] + trr_30z * dm_ij_cache[tx+48] + trr_40z * dm_ij_cache[tx+64];
                        double dot_lij_z_010 = trr_10z * dm_ij_cache[tx+96] + trr_20z * dm_ij_cache[tx+112] + trr_30z * dm_ij_cache[tx+128];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+144] + trr_10z * dm_ij_cache[tx+160] + trr_20z * dm_ij_cache[tx+176];
                        double dot_lij_z_030 = wt * dm_ij_cache[tx+192] + trr_10z * dm_ij_cache[tx+208];
                        double dot_lij_z_040 = wt * dm_ij_cache[tx+224];
                        double dot_lij_z_100 = trr_10z * dm_ij_cache[tx+256] + trr_20z * dm_ij_cache[tx+272] + trr_30z * dm_ij_cache[tx+288];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+304] + trr_10z * dm_ij_cache[tx+320] + trr_20z * dm_ij_cache[tx+336];
                        double dot_lij_z_120 = wt * dm_ij_cache[tx+352] + trr_10z * dm_ij_cache[tx+368];
                        double dot_lij_z_130 = wt * dm_ij_cache[tx+384];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+400] + trr_10z * dm_ij_cache[tx+416] + trr_20z * dm_ij_cache[tx+432];
                        double dot_lij_z_210 = wt * dm_ij_cache[tx+448] + trr_10z * dm_ij_cache[tx+464];
                        double dot_lij_z_220 = wt * dm_ij_cache[tx+480];
                        double dot_lij_z_300 = wt * dm_ij_cache[tx+496] + trr_10z * dm_ij_cache[tx+512];
                        double dot_lij_z_310 = wt * dm_ij_cache[tx+528];
                        double dot_lij_z_400 = wt * dm_ij_cache[tx+544];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030 + trr_40y * dot_lij_z_040;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120 + trr_30y * dot_lij_z_130;
                        double dot_lij_y_200 = dot_lij_z_200 + trr_10y * dot_lij_z_210 + trr_20y * dot_lij_z_220;
                        double dot_lij_y_300 = dot_lij_z_300 + trr_10y * dot_lij_z_310;
                        double dot_lij_y_400 = dot_lij_z_400;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        vj_kl_000 += dot_lij_y_000 + trr_10x * dot_lij_y_100 + trr_20x * dot_lij_y_200 + trr_30x * dot_lij_y_300 + trr_40x * dot_lij_y_400;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = wt * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_001 = trr_10z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_002 = trr_20z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_003 = trr_30z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_z_004 = trr_40z * dm_kl_cache[sq_kl+0];
                        double dot_lkl_y_000 = dot_lkl_z_000;
                        double dot_lkl_y_001 = dot_lkl_z_001;
                        double dot_lkl_y_002 = dot_lkl_z_002;
                        double dot_lkl_y_003 = dot_lkl_z_003;
                        double dot_lkl_y_004 = dot_lkl_z_004;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001;
                        double dot_lkl_y_012 = trr_10y * dot_lkl_z_002;
                        double dot_lkl_y_013 = trr_10y * dot_lkl_z_003;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000;
                        double dot_lkl_y_021 = trr_20y * dot_lkl_z_001;
                        double dot_lkl_y_022 = trr_20y * dot_lkl_z_002;
                        double dot_lkl_y_030 = trr_30y * dot_lkl_z_000;
                        double dot_lkl_y_031 = trr_30y * dot_lkl_z_001;
                        double dot_lkl_y_040 = trr_40y * dot_lkl_z_000;
                        vj_ij_002 += dot_lkl_y_002;
                        vj_ij_003 += dot_lkl_y_003;
                        vj_ij_004 += dot_lkl_y_004;
                        vj_ij_011 += dot_lkl_y_011;
                        vj_ij_012 += dot_lkl_y_012;
                        vj_ij_013 += dot_lkl_y_013;
                        vj_ij_020 += dot_lkl_y_020;
                        vj_ij_021 += dot_lkl_y_021;
                        vj_ij_022 += dot_lkl_y_022;
                        vj_ij_030 += dot_lkl_y_030;
                        vj_ij_031 += dot_lkl_y_031;
                        vj_ij_040 += dot_lkl_y_040;
                        vj_ij_101 += trr_10x * dot_lkl_y_001;
                        vj_ij_102 += trr_10x * dot_lkl_y_002;
                        vj_ij_103 += trr_10x * dot_lkl_y_003;
                        vj_ij_110 += trr_10x * dot_lkl_y_010;
                        vj_ij_111 += trr_10x * dot_lkl_y_011;
                        vj_ij_112 += trr_10x * dot_lkl_y_012;
                        vj_ij_120 += trr_10x * dot_lkl_y_020;
                        vj_ij_121 += trr_10x * dot_lkl_y_021;
                        vj_ij_130 += trr_10x * dot_lkl_y_030;
                        vj_ij_200 += trr_20x * dot_lkl_y_000;
                        vj_ij_201 += trr_20x * dot_lkl_y_001;
                        vj_ij_202 += trr_20x * dot_lkl_y_002;
                        vj_ij_210 += trr_20x * dot_lkl_y_010;
                        vj_ij_211 += trr_20x * dot_lkl_y_011;
                        vj_ij_220 += trr_20x * dot_lkl_y_020;
                        vj_ij_300 += trr_30x * dot_lkl_y_000;
                        vj_ij_301 += trr_30x * dot_lkl_y_001;
                        vj_ij_310 += trr_30x * dot_lkl_y_010;
                        vj_ij_400 += trr_40x * dot_lkl_y_000;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_000 += __shfl_down_sync(mask, vj_kl_000, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+0] += vj_kl_000;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_003;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_004;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+4, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_012;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_013;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_021;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+10, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_022;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+11, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_030;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+12, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_031;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+13, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_040;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+14, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+16, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_102;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+17, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_103;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+18, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+19, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_111;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+20, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_112;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+21, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_120;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+22, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_121;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+23, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_130;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+24, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+25, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_201;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+26, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_202;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+27, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_210;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+28, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_211;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+29, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_220;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+30, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_300;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+31, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_301;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+32, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_310;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+33, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_400;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+34, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=32, TILEY=27,
__global__
void rys_j_4_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 432;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 1728;
    double *dm_ij_cache = dm_kl_cache + 1728;
    double *rjri = dm_ij_cache + 560;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 108; n += 16) {
        int i = n / 27;
        int tile = n % 27;
        int task_kl = blockIdx.y * 432 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*432] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 1728; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_004 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_013 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_022 = 0;
        double vj_ij_030 = 0;
        double vj_ij_031 = 0;
        double vj_ij_040 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_103 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_112 = 0;
        double vj_ij_120 = 0;
        double vj_ij_121 = 0;
        double vj_ij_130 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_202 = 0;
        double vj_ij_210 = 0;
        double vj_ij_211 = 0;
        double vj_ij_220 = 0;
        double vj_ij_300 = 0;
        double vj_ij_301 = 0;
        double vj_ij_310 = 0;
        double vj_ij_400 = 0;
        for (int batch_kl = 0; batch_kl < 27; ++batch_kl) {
            int task_kl0 = blockIdx.y * 432 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*27+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_010 = 0;
            double vj_kl_100 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(3, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 3; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double dot_lij_z_000 = trr_20z * dm_ij_cache[tx+32] + trr_30z * dm_ij_cache[tx+48] + trr_40z * dm_ij_cache[tx+64];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double trr_41z = cpz * trr_40z + 4*b00 * trr_30z;
                        double dot_lij_z_001 = trr_21z * dm_ij_cache[tx+32] + trr_31z * dm_ij_cache[tx+48] + trr_41z * dm_ij_cache[tx+64];
                        double dot_lij_z_010 = trr_10z * dm_ij_cache[tx+96] + trr_20z * dm_ij_cache[tx+112] + trr_30z * dm_ij_cache[tx+128];
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double dot_lij_z_011 = trr_11z * dm_ij_cache[tx+96] + trr_21z * dm_ij_cache[tx+112] + trr_31z * dm_ij_cache[tx+128];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+144] + trr_10z * dm_ij_cache[tx+160] + trr_20z * dm_ij_cache[tx+176];
                        double trr_01z = cpz * wt;
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+144] + trr_11z * dm_ij_cache[tx+160] + trr_21z * dm_ij_cache[tx+176];
                        double dot_lij_z_030 = wt * dm_ij_cache[tx+192] + trr_10z * dm_ij_cache[tx+208];
                        double dot_lij_z_031 = trr_01z * dm_ij_cache[tx+192] + trr_11z * dm_ij_cache[tx+208];
                        double dot_lij_z_040 = wt * dm_ij_cache[tx+224];
                        double dot_lij_z_041 = trr_01z * dm_ij_cache[tx+224];
                        double dot_lij_z_100 = trr_10z * dm_ij_cache[tx+256] + trr_20z * dm_ij_cache[tx+272] + trr_30z * dm_ij_cache[tx+288];
                        double dot_lij_z_101 = trr_11z * dm_ij_cache[tx+256] + trr_21z * dm_ij_cache[tx+272] + trr_31z * dm_ij_cache[tx+288];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+304] + trr_10z * dm_ij_cache[tx+320] + trr_20z * dm_ij_cache[tx+336];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+304] + trr_11z * dm_ij_cache[tx+320] + trr_21z * dm_ij_cache[tx+336];
                        double dot_lij_z_120 = wt * dm_ij_cache[tx+352] + trr_10z * dm_ij_cache[tx+368];
                        double dot_lij_z_121 = trr_01z * dm_ij_cache[tx+352] + trr_11z * dm_ij_cache[tx+368];
                        double dot_lij_z_130 = wt * dm_ij_cache[tx+384];
                        double dot_lij_z_131 = trr_01z * dm_ij_cache[tx+384];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+400] + trr_10z * dm_ij_cache[tx+416] + trr_20z * dm_ij_cache[tx+432];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+400] + trr_11z * dm_ij_cache[tx+416] + trr_21z * dm_ij_cache[tx+432];
                        double dot_lij_z_210 = wt * dm_ij_cache[tx+448] + trr_10z * dm_ij_cache[tx+464];
                        double dot_lij_z_211 = trr_01z * dm_ij_cache[tx+448] + trr_11z * dm_ij_cache[tx+464];
                        double dot_lij_z_220 = wt * dm_ij_cache[tx+480];
                        double dot_lij_z_221 = trr_01z * dm_ij_cache[tx+480];
                        double dot_lij_z_300 = wt * dm_ij_cache[tx+496] + trr_10z * dm_ij_cache[tx+512];
                        double dot_lij_z_301 = trr_01z * dm_ij_cache[tx+496] + trr_11z * dm_ij_cache[tx+512];
                        double dot_lij_z_310 = wt * dm_ij_cache[tx+528];
                        double dot_lij_z_311 = trr_01z * dm_ij_cache[tx+528];
                        double dot_lij_z_400 = wt * dm_ij_cache[tx+544];
                        double dot_lij_z_401 = trr_01z * dm_ij_cache[tx+544];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030 + trr_40y * dot_lij_z_040;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031 + trr_40y * dot_lij_z_041;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double trr_41y = cpy * trr_40y + 4*b00 * trr_30y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030 + trr_41y * dot_lij_z_040;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120 + trr_30y * dot_lij_z_130;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121 + trr_30y * dot_lij_z_131;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120 + trr_31y * dot_lij_z_130;
                        double dot_lij_y_200 = dot_lij_z_200 + trr_10y * dot_lij_z_210 + trr_20y * dot_lij_z_220;
                        double dot_lij_y_201 = dot_lij_z_201 + trr_10y * dot_lij_z_211 + trr_20y * dot_lij_z_221;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210 + trr_21y * dot_lij_z_220;
                        double dot_lij_y_300 = dot_lij_z_300 + trr_10y * dot_lij_z_310;
                        double dot_lij_y_301 = dot_lij_z_301 + trr_10y * dot_lij_z_311;
                        double dot_lij_y_310 = trr_01y * dot_lij_z_300 + trr_11y * dot_lij_z_310;
                        double dot_lij_y_400 = dot_lij_z_400;
                        double dot_lij_y_401 = dot_lij_z_401;
                        double dot_lij_y_410 = trr_01y * dot_lij_z_400;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201 + trr_30x * dot_lij_y_301 + trr_40x * dot_lij_y_401;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210 + trr_30x * dot_lij_y_310 + trr_40x * dot_lij_y_410;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double trr_41x = cpx * trr_40x + 4*b00 * trr_30x;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200 + trr_31x * dot_lij_y_300 + trr_41x * dot_lij_y_400;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+432];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+432];
                        double dot_lkl_z_002 = trr_21z * dm_kl_cache[sq_kl+432];
                        double dot_lkl_z_003 = trr_31z * dm_kl_cache[sq_kl+432];
                        double dot_lkl_z_004 = trr_41z * dm_kl_cache[sq_kl+432];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_012 = trr_20z * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_013 = trr_30z * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_014 = trr_40z * dm_kl_cache[sq_kl+864];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_102 = trr_20z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_103 = trr_30z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_104 = trr_40z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012;
                        double dot_lkl_y_003 = dot_lkl_z_003 + trr_01y * dot_lkl_z_013;
                        double dot_lkl_y_004 = dot_lkl_z_004 + trr_01y * dot_lkl_z_014;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011;
                        double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012;
                        double dot_lkl_y_013 = trr_10y * dot_lkl_z_003 + trr_11y * dot_lkl_z_013;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010;
                        double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011;
                        double dot_lkl_y_022 = trr_20y * dot_lkl_z_002 + trr_21y * dot_lkl_z_012;
                        double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010;
                        double dot_lkl_y_031 = trr_30y * dot_lkl_z_001 + trr_31y * dot_lkl_z_011;
                        double dot_lkl_y_040 = trr_40y * dot_lkl_z_000 + trr_41y * dot_lkl_z_010;
                        double dot_lkl_y_100 = dot_lkl_z_100;
                        double dot_lkl_y_101 = dot_lkl_z_101;
                        double dot_lkl_y_102 = dot_lkl_z_102;
                        double dot_lkl_y_103 = dot_lkl_z_103;
                        double dot_lkl_y_104 = dot_lkl_z_104;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101;
                        double dot_lkl_y_112 = trr_10y * dot_lkl_z_102;
                        double dot_lkl_y_113 = trr_10y * dot_lkl_z_103;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100;
                        double dot_lkl_y_121 = trr_20y * dot_lkl_z_101;
                        double dot_lkl_y_122 = trr_20y * dot_lkl_z_102;
                        double dot_lkl_y_130 = trr_30y * dot_lkl_z_100;
                        double dot_lkl_y_131 = trr_30y * dot_lkl_z_101;
                        double dot_lkl_y_140 = trr_40y * dot_lkl_z_100;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102;
                        vj_ij_003 += dot_lkl_y_003 + trr_01x * dot_lkl_y_103;
                        vj_ij_004 += dot_lkl_y_004 + trr_01x * dot_lkl_y_104;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111;
                        vj_ij_012 += dot_lkl_y_012 + trr_01x * dot_lkl_y_112;
                        vj_ij_013 += dot_lkl_y_013 + trr_01x * dot_lkl_y_113;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120;
                        vj_ij_021 += dot_lkl_y_021 + trr_01x * dot_lkl_y_121;
                        vj_ij_022 += dot_lkl_y_022 + trr_01x * dot_lkl_y_122;
                        vj_ij_030 += dot_lkl_y_030 + trr_01x * dot_lkl_y_130;
                        vj_ij_031 += dot_lkl_y_031 + trr_01x * dot_lkl_y_131;
                        vj_ij_040 += dot_lkl_y_040 + trr_01x * dot_lkl_y_140;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101;
                        vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102;
                        vj_ij_103 += trr_10x * dot_lkl_y_003 + trr_11x * dot_lkl_y_103;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110;
                        vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111;
                        vj_ij_112 += trr_10x * dot_lkl_y_012 + trr_11x * dot_lkl_y_112;
                        vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120;
                        vj_ij_121 += trr_10x * dot_lkl_y_021 + trr_11x * dot_lkl_y_121;
                        vj_ij_130 += trr_10x * dot_lkl_y_030 + trr_11x * dot_lkl_y_130;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100;
                        vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101;
                        vj_ij_202 += trr_20x * dot_lkl_y_002 + trr_21x * dot_lkl_y_102;
                        vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110;
                        vj_ij_211 += trr_20x * dot_lkl_y_011 + trr_21x * dot_lkl_y_111;
                        vj_ij_220 += trr_20x * dot_lkl_y_020 + trr_21x * dot_lkl_y_120;
                        vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100;
                        vj_ij_301 += trr_30x * dot_lkl_y_001 + trr_31x * dot_lkl_y_101;
                        vj_ij_310 += trr_30x * dot_lkl_y_010 + trr_31x * dot_lkl_y_110;
                        vj_ij_400 += trr_40x * dot_lkl_y_000 + trr_41x * dot_lkl_y_100;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+432] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+864] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1296] += vj_kl_100;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_003;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_004;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+4, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_012;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_013;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_021;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+10, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_022;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+11, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_030;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+12, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_031;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+13, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_040;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+14, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+16, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_102;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+17, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_103;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+18, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+19, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_111;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+20, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_112;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+21, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_120;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+22, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_121;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+23, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_130;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+24, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+25, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_201;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+26, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_202;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+27, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_210;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+28, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_211;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+29, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_220;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+30, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_300;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+31, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_301;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+32, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_310;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+33, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_400;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+34, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 108; n += 16) {
        int i = n / 27;
        int tile = n % 27;
        int task_kl = blockIdx.y * 432 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*432]);
        }
    }
}

// TILEX=32, TILEY=9,
__global__
void rys_j_4_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 144;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *dm_kl_cache = vj_kl_cache + 1440;
    double *dm_ij_cache = dm_kl_cache + 1440;
    double *rjri = dm_ij_cache + 560;
    double *rlrk = rjri + 128;
    double *cicj_cache = rlrk + 128;
    double *rw = cicj_cache + iprim*jprim*16 + sq_id;
    double *vj_ij_cache = dm_ij_cache;
    double *aij_cache = rjri + 48;
    double *akl_cache = rlrk + 48;
    double *Rp = rjri + 64;
    double *Rq = rlrk + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = tx; n < 90; n += 16) {
        int i = n / 9;
        int tile = n % 9;
        int task_kl = blockIdx.y * 144 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*144] = dm[kl_loc0+i];
        }
    }
    for (int n = sq_id; n < 1440; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();
    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
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
        if (ty == 0) {
            rjri[tx+0] = xjxi;
            rjri[tx+16] = yjyi;
            rjri[tx+32] = zjzi;
        }
        for (int ij = ty; ij < iprim*jprim; ij += 16) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * rr_ij);
            cicj_cache[tx+ij*16] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_004 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_013 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_022 = 0;
        double vj_ij_030 = 0;
        double vj_ij_031 = 0;
        double vj_ij_040 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_103 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_112 = 0;
        double vj_ij_120 = 0;
        double vj_ij_121 = 0;
        double vj_ij_130 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_202 = 0;
        double vj_ij_210 = 0;
        double vj_ij_211 = 0;
        double vj_ij_220 = 0;
        double vj_ij_300 = 0;
        double vj_ij_301 = 0;
        double vj_ij_310 = 0;
        double vj_ij_400 = 0;
        for (int batch_kl = 0; batch_kl < 9; ++batch_kl) {
            int task_kl0 = blockIdx.y * 144 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*32+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*9+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (tx == 0) {
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                rlrk[ty+0] = xlxk;
                rlrk[ty+16] = ylyk;
                rlrk[ty+32] = zlzk;
            }
            double vj_kl_001 = 0;
            double vj_kl_002 = 0;
            double vj_kl_010 = 0;
            double vj_kl_011 = 0;
            double vj_kl_020 = 0;
            double vj_kl_100 = 0;
            double vj_kl_101 = 0;
            double vj_kl_110 = 0;
            double vj_kl_200 = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (tx == 0) {
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = expk[kp];
                    double al = expl[lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[ty+0];
                    double ylyk = rlrk[ty+16];
                    double zlzk = rlrk[ty+32];
                    double xkl = rk[0] + xlxk * al_akl;
                    double ykl = rk[1] + ylyk * al_akl;
                    double zkl = rk[2] + zlzk * al_akl;
                    double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                    double theta_kl = ak * al / akl;
                    double Kcd = exp(-theta_kl * rr_kl);
                    double ckcl = ck[kp] * cl[lp] * Kcd;
                    akl_cache[ty] = akl;
                    Rq[ty+0] = xkl;
                    Rq[ty+16] = ykl;
                    Rq[ty+32] = zkl;
                    Rq[ty+48] = ckcl;
                }
                int ijprim = iprim * jprim;
                for (int ijp = 0; ijp < ijprim; ++ijp) {
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = expi[ip];
                    double aj = expj[jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double xij = ri[0] + rjri[tx+0] * aj_aij;
                    double yij = ri[1] + rjri[tx+16] * aj_aij;
                    double zij = ri[2] + rjri[tx+32] * aj_aij;
                    __syncthreads();
                    if (ty == 0) {
                        aij_cache[tx] = aij;
                        Rp[tx+0] = xij;
                        Rp[tx+16] = yij;
                        Rp[tx+32] = zij;
                    }
                    double xpq = xij - Rq[ty+0];
                    double ypq = yij - Rq[ty+16];
                    double zpq = zij - Rq[ty+32];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    double akl = akl_cache[ty];
                    double theta = aij * akl / (aij + akl);
                    double omega = 0.;
                    rys_roots_rs(4, theta, rr, omega, rw, 256, 0, 1);
                    double cicj = cicj_cache[tx+ijp*16];
                    double ckcl = Rq[ty+48];
                    double fac = fac_sym * cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    __syncthreads();
                    for (int irys = 0; irys < 4; ++irys) {
                        rw[(2*irys+1)*256] *= fac;
                    }
                    __syncthreads();
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*256];
                        double rt = rw[ 2*irys   *256];
                        double rt_aa = rt / (aij_cache[tx] + akl_cache[ty]);
                        double b00 = .5 * rt_aa;
                        double zpq = Rp[tx+32] - Rq[ty+32];
                        double rt_aij = rt_aa * akl_cache[ty];
                        double b10 = .5/aij_cache[tx] * (1 - rt_aij);
                        double c0z = Rp[tx+32]-ri[2] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double dot_lij_z_000 = trr_20z * dm_ij_cache[tx+32] + trr_30z * dm_ij_cache[tx+48] + trr_40z * dm_ij_cache[tx+64];
                        double rt_akl = rt_aa * aij_cache[tx];
                        double b01 = .5/akl_cache[ty] * (1 - rt_akl);
                        double cpz = Rq[ty+32]-rk[2] + zpq*rt_akl;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double trr_41z = cpz * trr_40z + 4*b00 * trr_30z;
                        double dot_lij_z_001 = trr_21z * dm_ij_cache[tx+32] + trr_31z * dm_ij_cache[tx+48] + trr_41z * dm_ij_cache[tx+64];
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                        double trr_42z = cpz * trr_41z + 1*b01 * trr_40z + 4*b00 * trr_31z;
                        double dot_lij_z_002 = trr_22z * dm_ij_cache[tx+32] + trr_32z * dm_ij_cache[tx+48] + trr_42z * dm_ij_cache[tx+64];
                        double dot_lij_z_010 = trr_10z * dm_ij_cache[tx+96] + trr_20z * dm_ij_cache[tx+112] + trr_30z * dm_ij_cache[tx+128];
                        double dot_lij_z_011 = trr_11z * dm_ij_cache[tx+96] + trr_21z * dm_ij_cache[tx+112] + trr_31z * dm_ij_cache[tx+128];
                        double trr_01z = cpz * wt;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double dot_lij_z_012 = trr_12z * dm_ij_cache[tx+96] + trr_22z * dm_ij_cache[tx+112] + trr_32z * dm_ij_cache[tx+128];
                        double dot_lij_z_020 = wt * dm_ij_cache[tx+144] + trr_10z * dm_ij_cache[tx+160] + trr_20z * dm_ij_cache[tx+176];
                        double dot_lij_z_021 = trr_01z * dm_ij_cache[tx+144] + trr_11z * dm_ij_cache[tx+160] + trr_21z * dm_ij_cache[tx+176];
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double dot_lij_z_022 = trr_02z * dm_ij_cache[tx+144] + trr_12z * dm_ij_cache[tx+160] + trr_22z * dm_ij_cache[tx+176];
                        double dot_lij_z_030 = wt * dm_ij_cache[tx+192] + trr_10z * dm_ij_cache[tx+208];
                        double dot_lij_z_031 = trr_01z * dm_ij_cache[tx+192] + trr_11z * dm_ij_cache[tx+208];
                        double dot_lij_z_032 = trr_02z * dm_ij_cache[tx+192] + trr_12z * dm_ij_cache[tx+208];
                        double dot_lij_z_040 = wt * dm_ij_cache[tx+224];
                        double dot_lij_z_041 = trr_01z * dm_ij_cache[tx+224];
                        double dot_lij_z_042 = trr_02z * dm_ij_cache[tx+224];
                        double dot_lij_z_100 = trr_10z * dm_ij_cache[tx+256] + trr_20z * dm_ij_cache[tx+272] + trr_30z * dm_ij_cache[tx+288];
                        double dot_lij_z_101 = trr_11z * dm_ij_cache[tx+256] + trr_21z * dm_ij_cache[tx+272] + trr_31z * dm_ij_cache[tx+288];
                        double dot_lij_z_102 = trr_12z * dm_ij_cache[tx+256] + trr_22z * dm_ij_cache[tx+272] + trr_32z * dm_ij_cache[tx+288];
                        double dot_lij_z_110 = wt * dm_ij_cache[tx+304] + trr_10z * dm_ij_cache[tx+320] + trr_20z * dm_ij_cache[tx+336];
                        double dot_lij_z_111 = trr_01z * dm_ij_cache[tx+304] + trr_11z * dm_ij_cache[tx+320] + trr_21z * dm_ij_cache[tx+336];
                        double dot_lij_z_112 = trr_02z * dm_ij_cache[tx+304] + trr_12z * dm_ij_cache[tx+320] + trr_22z * dm_ij_cache[tx+336];
                        double dot_lij_z_120 = wt * dm_ij_cache[tx+352] + trr_10z * dm_ij_cache[tx+368];
                        double dot_lij_z_121 = trr_01z * dm_ij_cache[tx+352] + trr_11z * dm_ij_cache[tx+368];
                        double dot_lij_z_122 = trr_02z * dm_ij_cache[tx+352] + trr_12z * dm_ij_cache[tx+368];
                        double dot_lij_z_130 = wt * dm_ij_cache[tx+384];
                        double dot_lij_z_131 = trr_01z * dm_ij_cache[tx+384];
                        double dot_lij_z_132 = trr_02z * dm_ij_cache[tx+384];
                        double dot_lij_z_200 = wt * dm_ij_cache[tx+400] + trr_10z * dm_ij_cache[tx+416] + trr_20z * dm_ij_cache[tx+432];
                        double dot_lij_z_201 = trr_01z * dm_ij_cache[tx+400] + trr_11z * dm_ij_cache[tx+416] + trr_21z * dm_ij_cache[tx+432];
                        double dot_lij_z_202 = trr_02z * dm_ij_cache[tx+400] + trr_12z * dm_ij_cache[tx+416] + trr_22z * dm_ij_cache[tx+432];
                        double dot_lij_z_210 = wt * dm_ij_cache[tx+448] + trr_10z * dm_ij_cache[tx+464];
                        double dot_lij_z_211 = trr_01z * dm_ij_cache[tx+448] + trr_11z * dm_ij_cache[tx+464];
                        double dot_lij_z_212 = trr_02z * dm_ij_cache[tx+448] + trr_12z * dm_ij_cache[tx+464];
                        double dot_lij_z_220 = wt * dm_ij_cache[tx+480];
                        double dot_lij_z_221 = trr_01z * dm_ij_cache[tx+480];
                        double dot_lij_z_222 = trr_02z * dm_ij_cache[tx+480];
                        double dot_lij_z_300 = wt * dm_ij_cache[tx+496] + trr_10z * dm_ij_cache[tx+512];
                        double dot_lij_z_301 = trr_01z * dm_ij_cache[tx+496] + trr_11z * dm_ij_cache[tx+512];
                        double dot_lij_z_302 = trr_02z * dm_ij_cache[tx+496] + trr_12z * dm_ij_cache[tx+512];
                        double dot_lij_z_310 = wt * dm_ij_cache[tx+528];
                        double dot_lij_z_311 = trr_01z * dm_ij_cache[tx+528];
                        double dot_lij_z_312 = trr_02z * dm_ij_cache[tx+528];
                        double dot_lij_z_400 = wt * dm_ij_cache[tx+544];
                        double dot_lij_z_401 = trr_01z * dm_ij_cache[tx+544];
                        double dot_lij_z_402 = trr_02z * dm_ij_cache[tx+544];
                        double ypq = Rp[tx+16] - Rq[ty+16];
                        double c0y = Rp[tx+16]-ri[1] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double dot_lij_y_000 = dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030 + trr_40y * dot_lij_z_040;
                        double dot_lij_y_001 = dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031 + trr_40y * dot_lij_z_041;
                        double dot_lij_y_002 = dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022 + trr_30y * dot_lij_z_032 + trr_40y * dot_lij_z_042;
                        double cpy = Rq[ty+16]-rk[1] + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double trr_41y = cpy * trr_40y + 4*b00 * trr_30y;
                        double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030 + trr_41y * dot_lij_z_040;
                        double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021 + trr_31y * dot_lij_z_031 + trr_41y * dot_lij_z_041;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                        double trr_42y = cpy * trr_41y + 1*b01 * trr_40y + 4*b00 * trr_31y;
                        double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020 + trr_32y * dot_lij_z_030 + trr_42y * dot_lij_z_040;
                        double dot_lij_y_100 = dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120 + trr_30y * dot_lij_z_130;
                        double dot_lij_y_101 = dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121 + trr_30y * dot_lij_z_131;
                        double dot_lij_y_102 = dot_lij_z_102 + trr_10y * dot_lij_z_112 + trr_20y * dot_lij_z_122 + trr_30y * dot_lij_z_132;
                        double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120 + trr_31y * dot_lij_z_130;
                        double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111 + trr_21y * dot_lij_z_121 + trr_31y * dot_lij_z_131;
                        double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110 + trr_22y * dot_lij_z_120 + trr_32y * dot_lij_z_130;
                        double dot_lij_y_200 = dot_lij_z_200 + trr_10y * dot_lij_z_210 + trr_20y * dot_lij_z_220;
                        double dot_lij_y_201 = dot_lij_z_201 + trr_10y * dot_lij_z_211 + trr_20y * dot_lij_z_221;
                        double dot_lij_y_202 = dot_lij_z_202 + trr_10y * dot_lij_z_212 + trr_20y * dot_lij_z_222;
                        double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210 + trr_21y * dot_lij_z_220;
                        double dot_lij_y_211 = trr_01y * dot_lij_z_201 + trr_11y * dot_lij_z_211 + trr_21y * dot_lij_z_221;
                        double dot_lij_y_220 = trr_02y * dot_lij_z_200 + trr_12y * dot_lij_z_210 + trr_22y * dot_lij_z_220;
                        double dot_lij_y_300 = dot_lij_z_300 + trr_10y * dot_lij_z_310;
                        double dot_lij_y_301 = dot_lij_z_301 + trr_10y * dot_lij_z_311;
                        double dot_lij_y_302 = dot_lij_z_302 + trr_10y * dot_lij_z_312;
                        double dot_lij_y_310 = trr_01y * dot_lij_z_300 + trr_11y * dot_lij_z_310;
                        double dot_lij_y_311 = trr_01y * dot_lij_z_301 + trr_11y * dot_lij_z_311;
                        double dot_lij_y_320 = trr_02y * dot_lij_z_300 + trr_12y * dot_lij_z_310;
                        double dot_lij_y_400 = dot_lij_z_400;
                        double dot_lij_y_401 = dot_lij_z_401;
                        double dot_lij_y_402 = dot_lij_z_402;
                        double dot_lij_y_410 = trr_01y * dot_lij_z_400;
                        double dot_lij_y_411 = trr_01y * dot_lij_z_401;
                        double dot_lij_y_420 = trr_02y * dot_lij_z_400;
                        double xpq = Rp[tx+0] - Rq[ty+0];
                        double c0x = Rp[tx+0]-ri[0] - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        vj_kl_001 += dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201 + trr_30x * dot_lij_y_301 + trr_40x * dot_lij_y_401;
                        vj_kl_002 += dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202 + trr_30x * dot_lij_y_302 + trr_40x * dot_lij_y_402;
                        vj_kl_010 += dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210 + trr_30x * dot_lij_y_310 + trr_40x * dot_lij_y_410;
                        vj_kl_011 += dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211 + trr_30x * dot_lij_y_311 + trr_40x * dot_lij_y_411;
                        vj_kl_020 += dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220 + trr_30x * dot_lij_y_320 + trr_40x * dot_lij_y_420;
                        double cpx = Rq[ty+0]-rk[0] + xpq*rt_akl;
                        double trr_01x = cpx * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double trr_41x = cpx * trr_40x + 4*b00 * trr_30x;
                        vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200 + trr_31x * dot_lij_y_300 + trr_41x * dot_lij_y_400;
                        vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201 + trr_31x * dot_lij_y_301 + trr_41x * dot_lij_y_401;
                        vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210 + trr_31x * dot_lij_y_310 + trr_41x * dot_lij_y_410;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                        double trr_42x = cpx * trr_41x + 1*b01 * trr_40x + 4*b00 * trr_31x;
                        vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200 + trr_32x * dot_lij_y_300 + trr_42x * dot_lij_y_400;
                        int sq_kl = ty + batch_kl * 16;
                        double dot_lkl_z_000 = trr_01z * dm_kl_cache[sq_kl+144] + trr_02z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_001 = trr_11z * dm_kl_cache[sq_kl+144] + trr_12z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_002 = trr_21z * dm_kl_cache[sq_kl+144] + trr_22z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_003 = trr_31z * dm_kl_cache[sq_kl+144] + trr_32z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_004 = trr_41z * dm_kl_cache[sq_kl+144] + trr_42z * dm_kl_cache[sq_kl+288];
                        double dot_lkl_z_010 = wt * dm_kl_cache[sq_kl+432] + trr_01z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_011 = trr_10z * dm_kl_cache[sq_kl+432] + trr_11z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_012 = trr_20z * dm_kl_cache[sq_kl+432] + trr_21z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_013 = trr_30z * dm_kl_cache[sq_kl+432] + trr_31z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_014 = trr_40z * dm_kl_cache[sq_kl+432] + trr_41z * dm_kl_cache[sq_kl+576];
                        double dot_lkl_z_020 = wt * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_021 = trr_10z * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_022 = trr_20z * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_023 = trr_30z * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_024 = trr_40z * dm_kl_cache[sq_kl+720];
                        double dot_lkl_z_100 = wt * dm_kl_cache[sq_kl+864] + trr_01z * dm_kl_cache[sq_kl+1008];
                        double dot_lkl_z_101 = trr_10z * dm_kl_cache[sq_kl+864] + trr_11z * dm_kl_cache[sq_kl+1008];
                        double dot_lkl_z_102 = trr_20z * dm_kl_cache[sq_kl+864] + trr_21z * dm_kl_cache[sq_kl+1008];
                        double dot_lkl_z_103 = trr_30z * dm_kl_cache[sq_kl+864] + trr_31z * dm_kl_cache[sq_kl+1008];
                        double dot_lkl_z_104 = trr_40z * dm_kl_cache[sq_kl+864] + trr_41z * dm_kl_cache[sq_kl+1008];
                        double dot_lkl_z_110 = wt * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_111 = trr_10z * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_112 = trr_20z * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_113 = trr_30z * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_114 = trr_40z * dm_kl_cache[sq_kl+1152];
                        double dot_lkl_z_200 = wt * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_201 = trr_10z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_202 = trr_20z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_203 = trr_30z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_z_204 = trr_40z * dm_kl_cache[sq_kl+1296];
                        double dot_lkl_y_000 = dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020;
                        double dot_lkl_y_001 = dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021;
                        double dot_lkl_y_002 = dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022;
                        double dot_lkl_y_003 = dot_lkl_z_003 + trr_01y * dot_lkl_z_013 + trr_02y * dot_lkl_z_023;
                        double dot_lkl_y_004 = dot_lkl_z_004 + trr_01y * dot_lkl_z_014 + trr_02y * dot_lkl_z_024;
                        double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020;
                        double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021;
                        double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012 + trr_12y * dot_lkl_z_022;
                        double dot_lkl_y_013 = trr_10y * dot_lkl_z_003 + trr_11y * dot_lkl_z_013 + trr_12y * dot_lkl_z_023;
                        double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020;
                        double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011 + trr_22y * dot_lkl_z_021;
                        double dot_lkl_y_022 = trr_20y * dot_lkl_z_002 + trr_21y * dot_lkl_z_012 + trr_22y * dot_lkl_z_022;
                        double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010 + trr_32y * dot_lkl_z_020;
                        double dot_lkl_y_031 = trr_30y * dot_lkl_z_001 + trr_31y * dot_lkl_z_011 + trr_32y * dot_lkl_z_021;
                        double dot_lkl_y_040 = trr_40y * dot_lkl_z_000 + trr_41y * dot_lkl_z_010 + trr_42y * dot_lkl_z_020;
                        double dot_lkl_y_100 = dot_lkl_z_100 + trr_01y * dot_lkl_z_110;
                        double dot_lkl_y_101 = dot_lkl_z_101 + trr_01y * dot_lkl_z_111;
                        double dot_lkl_y_102 = dot_lkl_z_102 + trr_01y * dot_lkl_z_112;
                        double dot_lkl_y_103 = dot_lkl_z_103 + trr_01y * dot_lkl_z_113;
                        double dot_lkl_y_104 = dot_lkl_z_104 + trr_01y * dot_lkl_z_114;
                        double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110;
                        double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111;
                        double dot_lkl_y_112 = trr_10y * dot_lkl_z_102 + trr_11y * dot_lkl_z_112;
                        double dot_lkl_y_113 = trr_10y * dot_lkl_z_103 + trr_11y * dot_lkl_z_113;
                        double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110;
                        double dot_lkl_y_121 = trr_20y * dot_lkl_z_101 + trr_21y * dot_lkl_z_111;
                        double dot_lkl_y_122 = trr_20y * dot_lkl_z_102 + trr_21y * dot_lkl_z_112;
                        double dot_lkl_y_130 = trr_30y * dot_lkl_z_100 + trr_31y * dot_lkl_z_110;
                        double dot_lkl_y_131 = trr_30y * dot_lkl_z_101 + trr_31y * dot_lkl_z_111;
                        double dot_lkl_y_140 = trr_40y * dot_lkl_z_100 + trr_41y * dot_lkl_z_110;
                        double dot_lkl_y_200 = dot_lkl_z_200;
                        double dot_lkl_y_201 = dot_lkl_z_201;
                        double dot_lkl_y_202 = dot_lkl_z_202;
                        double dot_lkl_y_203 = dot_lkl_z_203;
                        double dot_lkl_y_204 = dot_lkl_z_204;
                        double dot_lkl_y_210 = trr_10y * dot_lkl_z_200;
                        double dot_lkl_y_211 = trr_10y * dot_lkl_z_201;
                        double dot_lkl_y_212 = trr_10y * dot_lkl_z_202;
                        double dot_lkl_y_213 = trr_10y * dot_lkl_z_203;
                        double dot_lkl_y_220 = trr_20y * dot_lkl_z_200;
                        double dot_lkl_y_221 = trr_20y * dot_lkl_z_201;
                        double dot_lkl_y_222 = trr_20y * dot_lkl_z_202;
                        double dot_lkl_y_230 = trr_30y * dot_lkl_z_200;
                        double dot_lkl_y_231 = trr_30y * dot_lkl_z_201;
                        double dot_lkl_y_240 = trr_40y * dot_lkl_z_200;
                        vj_ij_002 += dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202;
                        vj_ij_003 += dot_lkl_y_003 + trr_01x * dot_lkl_y_103 + trr_02x * dot_lkl_y_203;
                        vj_ij_004 += dot_lkl_y_004 + trr_01x * dot_lkl_y_104 + trr_02x * dot_lkl_y_204;
                        vj_ij_011 += dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211;
                        vj_ij_012 += dot_lkl_y_012 + trr_01x * dot_lkl_y_112 + trr_02x * dot_lkl_y_212;
                        vj_ij_013 += dot_lkl_y_013 + trr_01x * dot_lkl_y_113 + trr_02x * dot_lkl_y_213;
                        vj_ij_020 += dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220;
                        vj_ij_021 += dot_lkl_y_021 + trr_01x * dot_lkl_y_121 + trr_02x * dot_lkl_y_221;
                        vj_ij_022 += dot_lkl_y_022 + trr_01x * dot_lkl_y_122 + trr_02x * dot_lkl_y_222;
                        vj_ij_030 += dot_lkl_y_030 + trr_01x * dot_lkl_y_130 + trr_02x * dot_lkl_y_230;
                        vj_ij_031 += dot_lkl_y_031 + trr_01x * dot_lkl_y_131 + trr_02x * dot_lkl_y_231;
                        vj_ij_040 += dot_lkl_y_040 + trr_01x * dot_lkl_y_140 + trr_02x * dot_lkl_y_240;
                        vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201;
                        vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102 + trr_12x * dot_lkl_y_202;
                        vj_ij_103 += trr_10x * dot_lkl_y_003 + trr_11x * dot_lkl_y_103 + trr_12x * dot_lkl_y_203;
                        vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210;
                        vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111 + trr_12x * dot_lkl_y_211;
                        vj_ij_112 += trr_10x * dot_lkl_y_012 + trr_11x * dot_lkl_y_112 + trr_12x * dot_lkl_y_212;
                        vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120 + trr_12x * dot_lkl_y_220;
                        vj_ij_121 += trr_10x * dot_lkl_y_021 + trr_11x * dot_lkl_y_121 + trr_12x * dot_lkl_y_221;
                        vj_ij_130 += trr_10x * dot_lkl_y_030 + trr_11x * dot_lkl_y_130 + trr_12x * dot_lkl_y_230;
                        vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200;
                        vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101 + trr_22x * dot_lkl_y_201;
                        vj_ij_202 += trr_20x * dot_lkl_y_002 + trr_21x * dot_lkl_y_102 + trr_22x * dot_lkl_y_202;
                        vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110 + trr_22x * dot_lkl_y_210;
                        vj_ij_211 += trr_20x * dot_lkl_y_011 + trr_21x * dot_lkl_y_111 + trr_22x * dot_lkl_y_211;
                        vj_ij_220 += trr_20x * dot_lkl_y_020 + trr_21x * dot_lkl_y_120 + trr_22x * dot_lkl_y_220;
                        vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100 + trr_32x * dot_lkl_y_200;
                        vj_ij_301 += trr_30x * dot_lkl_y_001 + trr_31x * dot_lkl_y_101 + trr_32x * dot_lkl_y_201;
                        vj_ij_310 += trr_30x * dot_lkl_y_010 + trr_31x * dot_lkl_y_110 + trr_32x * dot_lkl_y_210;
                        vj_ij_400 += trr_40x * dot_lkl_y_000 + trr_41x * dot_lkl_y_100 + trr_42x * dot_lkl_y_200;
                    }
                }
            }
            
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_001 += __shfl_down_sync(mask, vj_kl_001, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+144] += vj_kl_001;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_002 += __shfl_down_sync(mask, vj_kl_002, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+288] += vj_kl_002;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_010 += __shfl_down_sync(mask, vj_kl_010, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+432] += vj_kl_010;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_011 += __shfl_down_sync(mask, vj_kl_011, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+576] += vj_kl_011;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_020 += __shfl_down_sync(mask, vj_kl_020, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+720] += vj_kl_020;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_100 += __shfl_down_sync(mask, vj_kl_100, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+864] += vj_kl_100;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_101 += __shfl_down_sync(mask, vj_kl_101, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1008] += vj_kl_101;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_110 += __shfl_down_sync(mask, vj_kl_110, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1152] += vj_kl_110;
            }

            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl_200 += __shfl_down_sync(mask, vj_kl_200, offset);
            }
            __syncthreads();
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                int sq_kl = ty + batch_kl * 16;
                vj_kl_cache[sq_kl+1296] += vj_kl_200;
            }
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_002;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+2, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_003;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+3, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_004;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+4, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_011;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+6, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_012;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+7, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_013;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+8, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_020;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+9, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_021;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+10, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_022;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+11, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_030;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+12, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_031;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+13, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_040;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+14, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_101;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+16, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_102;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+17, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_103;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+18, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_110;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+19, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_111;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+20, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_112;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+21, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_120;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+22, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_121;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+23, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_130;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+24, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_200;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+25, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_201;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+26, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_202;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+27, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_210;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+28, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_211;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+29, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_220;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+30, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_300;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+31, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_301;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+32, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_310;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+33, vj_ij_cache[sq_id]);
        }
        __syncthreads();
        vj_ij_cache[sq_id] = vj_ij_400;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (ty < stride) {
                vj_ij_cache[sq_id] += vj_ij_cache[sq_id + stride*16];
            }
        }
        __syncthreads();
        if (ty == 0 && task_ij0+tx < npairs_ij) {
            atomicAdd(vj+ij_loc0+34, vj_ij_cache[sq_id]);
        }
    }
    __syncthreads();
    for (int n = tx; n < 90; n += 16) {
        int i = n / 9;
        int tile = n % 9;
        int task_kl = blockIdx.y * 144 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*144]);
        }
    }
}

int rys_j_unrolled1(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int npairs_ij = bounds->npairs_ij;
    int npairs_kl = bounds->npairs_kl;
    int ijkl = lij*9 + lkl;
    switch (ijkl) {
    case 0: { // lij=0, lkl=0, tilex=32, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 511) / 512);
        int buflen = iprim*jprim*16 + 1808;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_0_0, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_0_0<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 9: { // lij=1, lkl=0, tilex=32, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 511) / 512);
        int buflen = iprim*jprim*16 + 1856;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_1_0, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_1_0<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 10: { // lij=1, lkl=1, tilex=32, tiley=11
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 175) / 176);
        int buflen = iprim*jprim*16 + 2752;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_1_1, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_1_1<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 11: { // lij=1, lkl=2, tilex=32, tiley=14
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 223) / 224);
        int buflen = iprim*jprim*16 + 5824;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_1_2, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_1_2<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 18: { // lij=2, lkl=0, tilex=32, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 511) / 512);
        int buflen = iprim*jprim*16 + 2464;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_2_0, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_2_0<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 19: { // lij=2, lkl=1, tilex=32, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 511) / 512);
        int buflen = iprim*jprim*16 + 5536;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_2_1, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_2_1<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 20: { // lij=2, lkl=2, tilex=32, tiley=12
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 191) / 192);
        int buflen = iprim*jprim*16 + 5792;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_2_2, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_2_2<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 21: { // lij=2, lkl=3, tilex=32, tiley=6
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 95) / 96);
        int buflen = iprim*jprim*16 + 5792;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_2_3, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_2_3<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 27: { // lij=3, lkl=0, tilex=32, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 511) / 512);
        int buflen = iprim*jprim*16 + 2624;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_3_0, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_3_0<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 28: { // lij=3, lkl=1, tilex=32, tiley=29
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 463) / 464);
        int buflen = iprim*jprim*16 + 5824;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_3_1, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_3_1<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 29: { // lij=3, lkl=2, tilex=32, tiley=11
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 175) / 176);
        int buflen = iprim*jprim*16 + 5632;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_3_2, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_3_2<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 30: { // lij=3, lkl=3, tilex=32, tiley=5
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 79) / 80);
        int buflen = iprim*jprim*16 + 5824;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_3_3, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_3_3<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 36: { // lij=4, lkl=0, tilex=32, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 511) / 512);
        int buflen = iprim*jprim*16 + 3376;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_4_0, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_4_0<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 37: { // lij=4, lkl=1, tilex=32, tiley=27
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 431) / 432);
        int buflen = iprim*jprim*16 + 5808;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_4_1, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_4_1<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 38: { // lij=4, lkl=2, tilex=32, tiley=9
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 143) / 144);
        int buflen = iprim*jprim*16 + 5744;
        if (buflen > 6144) {
            cudaFuncSetAttribute(rys_j_4_2, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        }
        rys_j_4_2<<<blocks, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    default: return 0;
    }
    return 1;
}
