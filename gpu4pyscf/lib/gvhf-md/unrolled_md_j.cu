#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/gamma_inc_unrolled.cu"


// TILEX=64, TILEY=25,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void md_j_0_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 1024;
    int task_kl0 = blockIdx.y * 400;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 256;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 1600;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 16;
    double *dm_ij_cache = vj_kl_cache + 400;
    double *dm_kl_cache = dm_ij_cache + 16;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 1664; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 400; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 400; n += 256) {
        int task_kl = blockIdx.y * 400 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+400] = ykl;
            Rq_cache[n+800] = zkl;
            Rq_cache[n+1200] = akl;
        }
    }

    for (int n = tx; n < 25; n += 16) {
        int i = n / 25;
        int tile = n % 25;
        int task_kl = blockIdx.y * 400 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*400] = dm[kl_loc0+i];
        }
    }

    for (int batch_ij = 0; batch_ij < 64; ++batch_ij) {
        int task_ij0 = blockIdx.x * 1024 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 1; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 25; ++batch_kl) {
            int task_kl0 = blockIdx.y * 400 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+400];
            double zkl = Rq_cache[sq_kl+800];
            double akl = Rq_cache[sq_kl+1200];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 0);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 0; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 1; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
    for (int n = tx; n < 25; n += 16) {
        int i = n / 25;
        int tile = n % 25;
        int task_kl = blockIdx.y * 400 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*400]);
        }
    }
}

// TILEX=64, TILEY=22,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void md_j_1_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 1024;
    int task_kl0 = blockIdx.y * 352;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 512;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 1408;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 64;
    double *dm_ij_cache = vj_kl_cache + 352;
    double *dm_kl_cache = dm_ij_cache + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 1472; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 352; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 352; n += 256) {
        int task_kl = blockIdx.y * 352 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+352] = ykl;
            Rq_cache[n+704] = zkl;
            Rq_cache[n+1056] = akl;
        }
    }

    for (int n = tx; n < 22; n += 16) {
        int i = n / 22;
        int tile = n % 22;
        int task_kl = blockIdx.y * 352 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*352] = dm[kl_loc0+i];
        }
    }

    for (int batch_ij = 0; batch_ij < 64; ++batch_ij) {
        int task_ij0 = blockIdx.x * 1024 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 22; ++batch_kl) {
            int task_kl0 = blockIdx.y * 352 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+352];
            double zkl = Rq_cache[sq_kl+704];
            double akl = Rq_cache[sq_kl+1056];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 1);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 1; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+32];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 4; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
    for (int n = tx; n < 22; n += 16) {
        int i = n / 22;
        int tile = n % 22;
        int task_kl = blockIdx.y * 352 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*352]);
        }
    }
}

// TILEX=64, TILEY=9,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void md_j_1_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 1024;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 768;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 576;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 64;
    double *dm_ij_cache = vj_kl_cache + 576;
    double *dm_kl_cache = dm_ij_cache + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 640; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 576; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 144; n += 256) {
        int task_kl = blockIdx.y * 144 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+144] = ykl;
            Rq_cache[n+288] = zkl;
            Rq_cache[n+432] = akl;
        }
    }

    for (int n = tx; n < 36; n += 16) {
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

    for (int batch_ij = 0; batch_ij < 64; ++batch_ij) {
        int task_ij0 = blockIdx.x * 1024 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 9; ++batch_kl) {
            int task_kl0 = blockIdx.y * 144 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+144];
            double zkl = Rq_cache[sq_kl+288];
            double akl = Rq_cache[sq_kl+432];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 2);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 2; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+32];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_0_1 * dm_ij_cache[tx+0];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+32];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+144] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+16];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl -= R_0_0_2_0 * dm_ij_cache[tx+32];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+288] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+32];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl -= R_0_2_0_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+432] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+144];
            vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+288];
            vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+432];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+144];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+288];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+432];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+144];
            vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+288];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+432];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+144];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+288];
            vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+432];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 4; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
    for (int n = tx; n < 36; n += 16) {
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

// TILEX=64, TILEY=12,
__global__
void md_j_1_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 1024;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1024;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 768;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 64;
    double *dm_ij_cache = vj_kl_cache + 1920;
    double *dm_kl_cache = dm_ij_cache + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 832; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 1920; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 192; n += 256) {
        int task_kl = blockIdx.y * 192 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+192] = ykl;
            Rq_cache[n+384] = zkl;
            Rq_cache[n+576] = akl;
        }
    }

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

    for (int batch_ij = 0; batch_ij < 64; ++batch_ij) {
        int task_ij0 = blockIdx.x * 1024 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 12; ++batch_kl) {
            int task_kl0 = blockIdx.y * 192 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+192];
            double zkl = Rq_cache[sq_kl+384];
            double akl = Rq_cache[sq_kl+576];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 3);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 3; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+32];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_0_1 * dm_ij_cache[tx+0];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+32];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+192] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+0];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl += R_0_0_0_3 * dm_ij_cache[tx+16];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl += R_0_0_1_2 * dm_ij_cache[tx+32];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl += R_0_1_0_2 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+384] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+16];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl -= R_0_0_2_0 * dm_ij_cache[tx+32];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+576] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+0];
            vj_kl += R_0_0_1_2 * dm_ij_cache[tx+16];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_0_2_1 * dm_ij_cache[tx+32];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+768] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+0];
            vj_kl += R_0_0_2_1 * dm_ij_cache[tx+16];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl += R_0_0_3_0 * dm_ij_cache[tx+32];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl += R_0_1_2_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+960] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+32];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl -= R_0_2_0_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1152] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+0];
            vj_kl += R_0_1_0_2 * dm_ij_cache[tx+16];
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+32];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_2_0_1 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1344] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+0];
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+16];
            vj_kl += R_0_1_2_0 * dm_ij_cache[tx+32];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl += R_0_2_1_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1536] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+0];
            vj_kl += R_0_2_0_1 * dm_ij_cache[tx+16];
            vj_kl += R_0_2_1_0 * dm_ij_cache[tx+32];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl += R_0_3_0_0 * dm_ij_cache[tx+48];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1728] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+192];
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+384];
            vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+576];
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+768];
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+960];
            vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+1152];
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+1344];
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+1536];
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+1728];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+192];
            vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+384];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+576];
            vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+768];
            vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+960];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+1152];
            vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+1344];
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+1536];
            vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+1728];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+192];
            vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+384];
            vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+576];
            vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+768];
            vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+960];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+1152];
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+1344];
            vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+1536];
            vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+1728];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+192];
            vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+384];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+576];
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+768];
            vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+960];
            vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+1152];
            vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+1344];
            vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+1536];
            vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+1728];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 4; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
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

// TILEX=42, TILEY=17,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void md_j_2_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 672;
    int task_kl0 = blockIdx.y * 272;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 768;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 1088;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 160;
    double *dm_ij_cache = vj_kl_cache + 272;
    double *dm_kl_cache = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 1152; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 272; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 272; n += 256) {
        int task_kl = blockIdx.y * 272 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+272] = ykl;
            Rq_cache[n+544] = zkl;
            Rq_cache[n+816] = akl;
        }
    }

    for (int n = tx; n < 17; n += 16) {
        int i = n / 17;
        int tile = n % 17;
        int task_kl = blockIdx.y * 272 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*272] = dm[kl_loc0+i];
        }
    }

    for (int batch_ij = 0; batch_ij < 42; ++batch_ij) {
        int task_ij0 = blockIdx.x * 672 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 17; ++batch_kl) {
            int task_kl0 = blockIdx.y * 272 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+272];
            double zkl = Rq_cache[sq_kl+544];
            double akl = Rq_cache[sq_kl+816];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 2);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 2; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+64] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+80] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+96] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+112] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+128] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+144] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 10; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
    for (int n = tx; n < 17; n += 16) {
        int i = n / 17;
        int tile = n % 17;
        int task_kl = blockIdx.y * 272 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*272]);
        }
    }
}

// TILEX=42, TILEY=7,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void md_j_2_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 672;
    int task_kl0 = blockIdx.y * 112;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1024;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 448;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 160;
    double *dm_ij_cache = vj_kl_cache + 448;
    double *dm_kl_cache = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 512; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 448; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 112; n += 256) {
        int task_kl = blockIdx.y * 112 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+112] = ykl;
            Rq_cache[n+224] = zkl;
            Rq_cache[n+336] = akl;
        }
    }

    for (int n = tx; n < 28; n += 16) {
        int i = n / 7;
        int tile = n % 7;
        int task_kl = blockIdx.y * 112 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*112] = dm[kl_loc0+i];
        }
    }

    for (int batch_ij = 0; batch_ij < 42; ++batch_ij) {
        int task_ij0 = blockIdx.x * 672 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 7; ++batch_kl) {
            int task_kl0 = blockIdx.y * 112 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+112];
            double zkl = Rq_cache[sq_kl+224];
            double akl = Rq_cache[sq_kl+336];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 3);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 3; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl -= R_0_0_1_2 * dm_ij_cache[tx+64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl -= R_0_1_0_2 * dm_ij_cache[tx+112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl -= R_0_2_0_1 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+112] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl -= R_0_0_2_1 * dm_ij_cache[tx+64];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl -= R_0_1_2_0 * dm_ij_cache[tx+128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl -= R_0_2_1_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+224] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl -= R_0_2_1_0 * dm_ij_cache[tx+128];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl -= R_0_3_0_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+336] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_3 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+64] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_0_3_0 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+80] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+96] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+112] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+128] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+112];
            vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+224];
            vj_ij -= R_0_3_0_0 * dm_kl_cache[sq_kl+336];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+144] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 10; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
    for (int n = tx; n < 28; n += 16) {
        int i = n / 7;
        int tile = n % 7;
        int task_kl = blockIdx.y * 112 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*112]);
        }
    }
}

// TILEX=42, TILEY=11,
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void md_j_2_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 672;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1280;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 704;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 160;
    double *dm_ij_cache = vj_kl_cache + 1760;
    double *dm_kl_cache = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 768; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 1760; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 176; n += 256) {
        int task_kl = blockIdx.y * 176 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+176] = ykl;
            Rq_cache[n+352] = zkl;
            Rq_cache[n+528] = akl;
        }
    }

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

    for (int batch_ij = 0; batch_ij < 42; ++batch_ij) {
        int task_ij0 = blockIdx.x * 672 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 11; ++batch_kl) {
            int task_kl0 = blockIdx.y * 176 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+176];
            double zkl = Rq_cache[sq_kl+352];
            double akl = Rq_cache[sq_kl+528];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 4);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 4; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl -= R_0_0_1_2 * dm_ij_cache[tx+64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl -= R_0_1_0_2 * dm_ij_cache[tx+112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl -= R_0_2_0_1 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+176] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+0];
            vj_kl += R_0_0_0_3 * dm_ij_cache[tx+16];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl += R_0_0_0_4 * dm_ij_cache[tx+32];
            vj_kl += R_0_0_1_2 * dm_ij_cache[tx+48];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl += R_0_0_1_3 * dm_ij_cache[tx+64];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl += R_0_0_2_2 * dm_ij_cache[tx+80];
            vj_kl += R_0_1_0_2 * dm_ij_cache[tx+96];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl += R_0_1_0_3 * dm_ij_cache[tx+112];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl += R_0_1_1_2 * dm_ij_cache[tx+128];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl += R_0_2_0_2 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+352] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl -= R_0_0_2_1 * dm_ij_cache[tx+64];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl -= R_0_1_2_0 * dm_ij_cache[tx+128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl -= R_0_2_1_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+528] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+0];
            vj_kl += R_0_0_1_2 * dm_ij_cache[tx+16];
            vj_kl += R_0_0_1_3 * dm_ij_cache[tx+32];
            vj_kl += R_0_0_2_1 * dm_ij_cache[tx+48];
            vj_kl += R_0_0_2_2 * dm_ij_cache[tx+64];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl += R_0_0_3_1 * dm_ij_cache[tx+80];
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+96];
            vj_kl += R_0_1_1_2 * dm_ij_cache[tx+112];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl += R_0_1_2_1 * dm_ij_cache[tx+128];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl += R_0_2_1_1 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+704] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+0];
            vj_kl += R_0_0_2_1 * dm_ij_cache[tx+16];
            vj_kl += R_0_0_2_2 * dm_ij_cache[tx+32];
            vj_kl += R_0_0_3_0 * dm_ij_cache[tx+48];
            vj_kl += R_0_0_3_1 * dm_ij_cache[tx+64];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl += R_0_0_4_0 * dm_ij_cache[tx+80];
            vj_kl += R_0_1_2_0 * dm_ij_cache[tx+96];
            vj_kl += R_0_1_2_1 * dm_ij_cache[tx+112];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl += R_0_1_3_0 * dm_ij_cache[tx+128];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl += R_0_2_2_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+880] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl -= R_0_2_1_0 * dm_ij_cache[tx+128];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl -= R_0_3_0_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1056] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+0];
            vj_kl += R_0_1_0_2 * dm_ij_cache[tx+16];
            vj_kl += R_0_1_0_3 * dm_ij_cache[tx+32];
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+48];
            vj_kl += R_0_1_1_2 * dm_ij_cache[tx+64];
            vj_kl += R_0_1_2_1 * dm_ij_cache[tx+80];
            vj_kl += R_0_2_0_1 * dm_ij_cache[tx+96];
            vj_kl += R_0_2_0_2 * dm_ij_cache[tx+112];
            vj_kl += R_0_2_1_1 * dm_ij_cache[tx+128];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl += R_0_3_0_1 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1232] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+0];
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+16];
            vj_kl += R_0_1_1_2 * dm_ij_cache[tx+32];
            vj_kl += R_0_1_2_0 * dm_ij_cache[tx+48];
            vj_kl += R_0_1_2_1 * dm_ij_cache[tx+64];
            vj_kl += R_0_1_3_0 * dm_ij_cache[tx+80];
            vj_kl += R_0_2_1_0 * dm_ij_cache[tx+96];
            vj_kl += R_0_2_1_1 * dm_ij_cache[tx+112];
            vj_kl += R_0_2_2_0 * dm_ij_cache[tx+128];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl += R_0_3_1_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1408] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+0];
            vj_kl += R_0_2_0_1 * dm_ij_cache[tx+16];
            vj_kl += R_0_2_0_2 * dm_ij_cache[tx+32];
            vj_kl += R_0_2_1_0 * dm_ij_cache[tx+48];
            vj_kl += R_0_2_1_1 * dm_ij_cache[tx+64];
            vj_kl += R_0_2_2_0 * dm_ij_cache[tx+80];
            vj_kl += R_0_3_0_0 * dm_ij_cache[tx+96];
            vj_kl += R_0_3_0_1 * dm_ij_cache[tx+112];
            vj_kl += R_0_3_1_0 * dm_ij_cache[tx+128];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl += R_0_4_0_0 * dm_ij_cache[tx+144];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+1584] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_3 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_0_0_4 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_0_1_3 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_0_2_2 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_1_0_3 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_2_0_2 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_0_1_3 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_0_2_2 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_0_3_1 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+64] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_0_2_2 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_0_3_0 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_0_3_1 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_0_4_0 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_1_3_0 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_2_2_0 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+80] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+96] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_1_0_3 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_2_0_2 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_3_0_1 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+112] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_1_3_0 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_2_2_0 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_3_1_0 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+128] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+176];
            vj_ij += R_0_2_0_2 * dm_kl_cache[sq_kl+352];
            vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+528];
            vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+704];
            vj_ij += R_0_2_2_0 * dm_kl_cache[sq_kl+880];
            vj_ij -= R_0_3_0_0 * dm_kl_cache[sq_kl+1056];
            vj_ij += R_0_3_0_1 * dm_kl_cache[sq_kl+1232];
            vj_ij += R_0_3_1_0 * dm_kl_cache[sq_kl+1408];
            vj_ij += R_0_4_0_0 * dm_kl_cache[sq_kl+1584];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+144] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 10; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
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

// TILEX=32, TILEY=32,
__global__
void md_j_3_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1024;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 2048;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 320;
    double *dm_ij_cache = vj_kl_cache + 512;
    double *dm_kl_cache = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 2112; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 512; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 512; n += 256) {
        int task_kl = blockIdx.y * 512 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+512] = ykl;
            Rq_cache[n+1024] = zkl;
            Rq_cache[n+1536] = akl;
        }
    }

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

    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+512];
            double zkl = Rq_cache[sq_kl+1024];
            double akl = Rq_cache[sq_kl+1536];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 3);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 3; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl += R_0_0_1_2 * dm_ij_cache[tx+96];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_0_2_1 * dm_ij_cache[tx+128];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl += R_0_0_3_0 * dm_ij_cache[tx+144];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl += R_0_1_0_2 * dm_ij_cache[tx+192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl += R_0_1_2_0 * dm_ij_cache[tx+240];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_2_0_1 * dm_ij_cache[tx+272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl += R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl += R_0_3_0_0 * dm_ij_cache[tx+304];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+64] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+80] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+96] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+112] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+128] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+144] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+160] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+176] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+192] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+208] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+224] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+240] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+256] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+272] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+288] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+304] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 20; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
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

// TILEX=32, TILEY=20,
__global__
void md_j_3_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 320;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1280;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 1280;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 320;
    double *dm_ij_cache = vj_kl_cache + 1280;
    double *dm_kl_cache = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 1344; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 1280; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 320; n += 256) {
        int task_kl = blockIdx.y * 320 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+320] = ykl;
            Rq_cache[n+640] = zkl;
            Rq_cache[n+960] = akl;
        }
    }

    for (int n = tx; n < 80; n += 16) {
        int i = n / 20;
        int tile = n % 20;
        int task_kl = blockIdx.y * 320 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*320] = dm[kl_loc0+i];
        }
    }

    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 20; ++batch_kl) {
            int task_kl0 = blockIdx.y * 320 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+320];
            double zkl = Rq_cache[sq_kl+640];
            double akl = Rq_cache[sq_kl+960];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 4);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 4; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl += R_0_0_1_2 * dm_ij_cache[tx+96];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_0_2_1 * dm_ij_cache[tx+128];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl += R_0_0_3_0 * dm_ij_cache[tx+144];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl += R_0_1_0_2 * dm_ij_cache[tx+192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl += R_0_1_2_0 * dm_ij_cache[tx+240];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_2_0_1 * dm_ij_cache[tx+272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl += R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl += R_0_3_0_0 * dm_ij_cache[tx+304];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl -= R_0_0_0_3 * dm_ij_cache[tx+32];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl -= R_0_0_0_4 * dm_ij_cache[tx+48];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl -= R_0_0_1_2 * dm_ij_cache[tx+80];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl -= R_0_0_1_3 * dm_ij_cache[tx+96];
            vj_kl -= R_0_0_2_1 * dm_ij_cache[tx+112];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl -= R_0_0_2_2 * dm_ij_cache[tx+128];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl -= R_0_0_3_1 * dm_ij_cache[tx+144];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl -= R_0_1_0_2 * dm_ij_cache[tx+176];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl -= R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+208];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl -= R_0_1_1_2 * dm_ij_cache[tx+224];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl -= R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl -= R_0_2_0_1 * dm_ij_cache[tx+256];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl -= R_0_2_0_2 * dm_ij_cache[tx+272];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl -= R_0_2_1_1 * dm_ij_cache[tx+288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl -= R_0_3_0_1 * dm_ij_cache[tx+304];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+320] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl -= R_0_0_1_3 * dm_ij_cache[tx+48];
            vj_kl -= R_0_0_2_0 * dm_ij_cache[tx+64];
            vj_kl -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl -= R_0_0_2_2 * dm_ij_cache[tx+96];
            vj_kl -= R_0_0_3_0 * dm_ij_cache[tx+112];
            vj_kl -= R_0_0_3_1 * dm_ij_cache[tx+128];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl -= R_0_0_4_0 * dm_ij_cache[tx+144];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl -= R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl -= R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl -= R_0_1_2_1 * dm_ij_cache[tx+224];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl -= R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl -= R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl -= R_0_2_1_1 * dm_ij_cache[tx+272];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl -= R_0_2_2_0 * dm_ij_cache[tx+288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl -= R_0_3_1_0 * dm_ij_cache[tx+304];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+640] += vj_kl;
            }
            vj_kl = 0.;
            vj_kl -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl -= R_0_1_0_3 * dm_ij_cache[tx+48];
            vj_kl -= R_0_1_1_0 * dm_ij_cache[tx+64];
            vj_kl -= R_0_1_1_1 * dm_ij_cache[tx+80];
            vj_kl -= R_0_1_1_2 * dm_ij_cache[tx+96];
            vj_kl -= R_0_1_2_0 * dm_ij_cache[tx+112];
            vj_kl -= R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl -= R_0_1_3_0 * dm_ij_cache[tx+144];
            vj_kl -= R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl -= R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl -= R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl -= R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl -= R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl -= R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl -= R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl -= R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl -= R_0_3_1_0 * dm_ij_cache[tx+288];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl -= R_0_4_0_0 * dm_ij_cache[tx+304];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+960] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_3 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_0_4 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_1_3 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_0_3 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+64] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+80] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_1_3 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_2_2 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_1_2 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+96] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_3_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+112] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_2_2 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_3_1 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_2_1 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+128] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_0_3_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_0_4_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_1_3_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+144] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+160] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+176] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_0_3 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_1_1_2 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_2_0_2 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+192] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+208] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_1_2 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_1_2_1 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_2_1_1 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+224] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_1_2_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_1_3_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_2_2_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+240] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_3_0_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+256] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_2_0_2 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_2_1_1 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_3_0_1 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+272] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_2_1_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_2_2_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_3_1_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+288] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+0];
            vj_ij -= R_0_3_0_1 * dm_kl_cache[sq_kl+320];
            vj_ij -= R_0_3_1_0 * dm_kl_cache[sq_kl+640];
            vj_ij -= R_0_4_0_0 * dm_kl_cache[sq_kl+960];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+304] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 20; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
    for (int n = tx; n < 80; n += 16) {
        int i = n / 20;
        int tile = n % 20;
        int task_kl = blockIdx.y * 320 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int kl_loc0 = pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+kl_loc0+i, vj_kl_cache[sq_kl+i*320]);
        }
    }
}

// TILEX=25, TILEY=32,
__global__
void md_j_4_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 400;
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
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1280;
    double *Rq_cache = Rp_cache + 64;
    double *vj_cache = Rq_cache + 2048;
    double *vj_ij_cache = vj_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 560;
    double *dm_ij_cache = vj_kl_cache + 512;
    double *dm_kl_cache = dm_ij_cache + 560;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    // zero out all cache;
    for (int n = sq_id; n < 2112; n += 256) {
        Rp_cache[n] = 1.;
    }
    for (int n = sq_id; n < 512; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 512; n += 256) {
        int task_kl = blockIdx.y * 512 + n;
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
            Rq_cache[n+0] = xkl;
            Rq_cache[n+512] = ykl;
            Rq_cache[n+1024] = zkl;
            Rq_cache[n+1536] = akl;
        }
    }

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

    for (int batch_ij = 0; batch_ij < 25; ++batch_ij) {
        int task_ij0 = blockIdx.x * 400 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        for (int n = sq_id; n < 16; n += 256) {
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
                Rp_cache[n+0] = xij;
                Rp_cache[n+16] = yij;
                Rp_cache[n+32] = zij;
                Rp_cache[n+48] = aij;
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
        int ij_loc0 = pair_loc[pair_ij];
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
            vj_ij_cache[tx+n*16] = 0;
        }
        for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
            int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[blockIdx.x*+batch_ij] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[blockIdx.y*+batch_kl] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
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
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+512];
            double zkl = Rq_cache[sq_kl+1024];
            double akl = Rq_cache[sq_kl+1536];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 4);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 4; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl = 0.;
            vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl += R_0_0_0_4 * dm_ij_cache[tx+64];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_1_0 * dm_ij_cache[tx+80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl += R_0_0_1_1 * dm_ij_cache[tx+96];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl += R_0_0_1_2 * dm_ij_cache[tx+112];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl += R_0_0_1_3 * dm_ij_cache[tx+128];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_0_2_0 * dm_ij_cache[tx+144];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_0_2_1 * dm_ij_cache[tx+160];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl += R_0_0_2_2 * dm_ij_cache[tx+176];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl += R_0_0_3_0 * dm_ij_cache[tx+192];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl += R_0_0_3_1 * dm_ij_cache[tx+208];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl += R_0_0_4_0 * dm_ij_cache[tx+224];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl += R_0_1_0_0 * dm_ij_cache[tx+240];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl += R_0_1_0_1 * dm_ij_cache[tx+256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl += R_0_1_0_2 * dm_ij_cache[tx+272];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl += R_0_1_0_3 * dm_ij_cache[tx+288];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl += R_0_1_1_0 * dm_ij_cache[tx+304];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl += R_0_1_1_1 * dm_ij_cache[tx+320];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl += R_0_1_1_2 * dm_ij_cache[tx+336];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl += R_0_1_2_0 * dm_ij_cache[tx+352];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl += R_0_1_2_1 * dm_ij_cache[tx+368];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl += R_0_1_3_0 * dm_ij_cache[tx+384];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl += R_0_2_0_0 * dm_ij_cache[tx+400];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl += R_0_2_0_1 * dm_ij_cache[tx+416];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl += R_0_2_0_2 * dm_ij_cache[tx+432];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl += R_0_2_1_0 * dm_ij_cache[tx+448];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl += R_0_2_1_1 * dm_ij_cache[tx+464];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl += R_0_2_2_0 * dm_ij_cache[tx+480];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl += R_0_3_0_0 * dm_ij_cache[tx+496];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl += R_0_3_0_1 * dm_ij_cache[tx+512];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl += R_0_3_1_0 * dm_ij_cache[tx+528];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl += R_0_4_0_0 * dm_ij_cache[tx+544];

            for (int offset = 8.0; offset > 0; offset /= 2) {
                vj_kl += __shfl_down_sync(mask, vj_kl, offset);
            }
            if (tx == 0 && task_kl0+ty < npairs_kl) {
                vj_kl_cache[sq_kl+0] += vj_kl;
            }
            vj_ij = 0.;
            vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+0] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+16] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+32] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+48] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_0_4 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+64] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+80] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+96] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+112] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_1_3 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+128] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+144] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+160] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_2_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+176] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+192] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_3_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+208] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_0_4_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+224] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+240] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+256] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+272] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_0_3 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+288] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+304] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+320] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+336] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+352] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+368] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_1_3_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+384] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+400] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+416] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_0_2 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+432] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+448] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+464] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_2_2_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+480] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+496] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_3_0_1 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+512] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_3_1_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+528] += vj_cache[sq_id];
            }
            vj_ij = 0.;
            vj_ij += R_0_4_0_0 * dm_kl_cache[sq_kl+0];
            __syncthreads();
            vj_cache[sq_id] = vj_ij;
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[sq_id] += vj_cache[sq_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                vj_ij_cache[tx+544] += vj_cache[sq_id];
            }
            __syncthreads();
        }
        // The last tile for ij
        if (task_ij0+tx < npairs_ij) {
            int ij_loc0 = pair_loc[pair_ij];
            for (int n = ty; n < 35; n += 16) {
                atomicAdd(vj+ij_loc0+n, vj_ij_cache[tx+n*16]);
            }
        }
    }
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

int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int ijkl = lij*9 + lkl;
    int npairs_ij = bounds->npairs_ij;
    int npairs_kl = bounds->npairs_kl;
    switch (ijkl) {
    case 0: { // lij=0, lkl=0, tilex=64, tiley=25
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 1023) / 1024, (npairs_kl + 399) / 400);
        md_j_0_0<<<blocks, threads, 3008*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 9: { // lij=1, lkl=0, tilex=64, tiley=22
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 1023) / 1024, (npairs_kl + 351) / 352);
        md_j_1_0<<<blocks, threads, 3072*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 10: { // lij=1, lkl=1, tilex=64, tiley=9
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 1023) / 1024, (npairs_kl + 143) / 144);
        md_j_1_1<<<blocks, threads, 2944*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 11: { // lij=1, lkl=2, tilex=64, tiley=12
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 1023) / 1024, (npairs_kl + 191) / 192);
        md_j_1_2<<<blocks, threads, 6080*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 18: { // lij=2, lkl=0, tilex=42, tiley=17
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 671) / 672, (npairs_kl + 271) / 272);
        md_j_2_0<<<blocks, threads, 3040*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 19: { // lij=2, lkl=1, tilex=42, tiley=7
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 671) / 672, (npairs_kl + 111) / 112);
        md_j_2_1<<<blocks, threads, 3008*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 20: { // lij=2, lkl=2, tilex=42, tiley=11
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 671) / 672, (npairs_kl + 175) / 176);
        md_j_2_2<<<blocks, threads, 6144*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 27: { // lij=3, lkl=0, tilex=32, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 511) / 512);
        md_j_3_0<<<blocks, threads, 5056*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 28: { // lij=3, lkl=1, tilex=32, tiley=20
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 511) / 512, (npairs_kl + 319) / 320);
        md_j_3_1<<<blocks, threads, 6080*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 36: { // lij=4, lkl=0, tilex=25, tiley=32
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 399) / 400, (npairs_kl + 511) / 512);
        md_j_4_0<<<blocks, threads, 5792*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    default: return 0;
    }
    return 1;
}
