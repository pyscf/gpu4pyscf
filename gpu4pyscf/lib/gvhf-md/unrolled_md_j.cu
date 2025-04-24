#include <stdio.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/gamma_inc_unrolled.cu"


// TILEX=32, TILEY=16, cache_dm=True
__global__
void md_j_0_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 512;
    int task_kl0 = blockIdx.y * 256;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 256;
    double *Rq_cache = Rp_cache + 2048;
    double *vj_ij_cache = Rq_cache + 1024;
    double *vj_kl_cache = vj_ij_cache + 512;
    double *vj_cache = vj_kl_cache + 256;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 512;
    // zero out all cache;
    for (int n = sq_id; n < 4864; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 512; n += 256) {
        int task_ij = blockIdx.x * 512 + n;
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
            Rp_cache[n+512] = yij;
            Rp_cache[n+1024] = zij;
            Rp_cache[n+1536] = aij;
        } else {
            Rp_cache[n+1536] = 1.;
        }
    }
    for (int n = sq_id; n < 256; n += 256) {
        int task_kl = blockIdx.y * 256 + n;
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
            Rq_cache[n+256] = ykl;
            Rq_cache[n+512] = zkl;
            Rq_cache[n+768] = akl;
        } else {
            Rq_cache[n+768] = 1.;
        }
    }
    for (int n = ty; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_ij = blockIdx.x * 512 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*512] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 16; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*256] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 32; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 16; ++batch_kl) {
        int task_ij0 = blockIdx.x * 512 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 256 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+512];
        double zij = Rp_cache[sq_ij+1024];
        double aij = Rp_cache[sq_ij+1536];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+256];
        double zkl = Rq_cache[sq_kl+512];
        double akl = Rq_cache[sq_kl+768];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_ij = blockIdx.x * 512 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*512]);
        }
    }
    for (int n = tx; n < 16; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*256]);
        }
    }
}

// TILEX=8, TILEY=32, cache_dm=True
__global__
void md_j_1_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 128;
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
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 512;
    double *Rq_cache = Rp_cache + 512;
    double *vj_ij_cache = Rq_cache + 2048;
    double *vj_kl_cache = vj_ij_cache + 512;
    double *vj_cache = vj_kl_cache + 512;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 512;
    // zero out all cache;
    for (int n = sq_id; n < 4864; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 128; n += 256) {
        int task_ij = blockIdx.x * 128 + n;
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
            Rp_cache[n+128] = yij;
            Rp_cache[n+256] = zij;
            Rp_cache[n+384] = aij;
        } else {
            Rp_cache[n+384] = 1.;
        }
    }
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
        } else {
            Rq_cache[n+1536] = 1.;
        }
    }
    for (int n = ty; n < 32; n += 16) {
        int i = n / 8;
        int tile = n % 8;
        int task_ij = blockIdx.x * 128 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*128] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 8; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
        int task_ij0 = blockIdx.x * 128 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+128];
        double zij = Rp_cache[sq_ij+256];
        double aij = Rp_cache[sq_ij+384];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+512];
        double zkl = Rq_cache[sq_kl+1024];
        double akl = Rq_cache[sq_kl+1536];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+128];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+256];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+384];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+128] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+384] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 32; n += 16) {
        int i = n / 8;
        int tile = n % 8;
        int task_ij = blockIdx.x * 128 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*128]);
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=16, TILEY=8, cache_dm=True
__global__
void md_j_1_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 256;
    int task_kl0 = blockIdx.y * 128;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 768;
    double *Rq_cache = Rp_cache + 1024;
    double *vj_ij_cache = Rq_cache + 512;
    double *vj_kl_cache = vj_ij_cache + 1024;
    double *vj_cache = vj_kl_cache + 512;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 1024;
    // zero out all cache;
    for (int n = sq_id; n < 4864; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 256; n += 256) {
        int task_ij = blockIdx.x * 256 + n;
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
            Rp_cache[n+256] = yij;
            Rp_cache[n+512] = zij;
            Rp_cache[n+768] = aij;
        } else {
            Rp_cache[n+768] = 1.;
        }
    }
    for (int n = sq_id; n < 128; n += 256) {
        int task_kl = blockIdx.y * 128 + n;
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
            Rq_cache[n+128] = ykl;
            Rq_cache[n+256] = zkl;
            Rq_cache[n+384] = akl;
        } else {
            Rq_cache[n+384] = 1.;
        }
    }
    for (int n = ty; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_ij = blockIdx.x * 256 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*256] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 8;
        int tile = n % 8;
        int task_kl = blockIdx.y * 128 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*128] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 16; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 8; ++batch_kl) {
        int task_ij0 = blockIdx.x * 256 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 128 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+256];
        double zij = Rp_cache[sq_ij+512];
        double aij = Rp_cache[sq_ij+768];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+128];
        double zkl = Rq_cache[sq_kl+256];
        double akl = Rq_cache[sq_kl+384];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+256];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+512];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_0_1 * dm_ij_cache[sq_ij+0];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl -= R_0_0_0_2 * dm_ij_cache[sq_ij+256];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+512];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+128] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_1_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+256];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl -= R_0_0_2_0 * dm_ij_cache[sq_ij+512];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+256] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_1_0_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+256];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl -= R_0_2_0_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+384] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+384];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+384];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+384];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+384];
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
            vj_ij_cache[sq_ij+768] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_ij = blockIdx.x * 256 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*256]);
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 8;
        int tile = n % 8;
        int task_kl = blockIdx.y * 128 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*128]);
        }
    }
}

// TILEX=16, TILEY=4, cache_dm=True
__global__
void md_j_1_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 256;
    int task_kl0 = blockIdx.y * 64;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1024;
    double *Rq_cache = Rp_cache + 1024;
    double *vj_ij_cache = Rq_cache + 256;
    double *vj_kl_cache = vj_ij_cache + 1024;
    double *vj_cache = vj_kl_cache + 640;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 1024;
    // zero out all cache;
    for (int n = sq_id; n < 4864; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 256; n += 256) {
        int task_ij = blockIdx.x * 256 + n;
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
            Rp_cache[n+256] = yij;
            Rp_cache[n+512] = zij;
            Rp_cache[n+768] = aij;
        } else {
            Rp_cache[n+768] = 1.;
        }
    }
    for (int n = sq_id; n < 64; n += 256) {
        int task_kl = blockIdx.y * 64 + n;
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
            Rq_cache[n+64] = ykl;
            Rq_cache[n+128] = zkl;
            Rq_cache[n+192] = akl;
        } else {
            Rq_cache[n+192] = 1.;
        }
    }
    for (int n = ty; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_ij = blockIdx.x * 256 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*256] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 40; n += 16) {
        int i = n / 4;
        int tile = n % 4;
        int task_kl = blockIdx.y * 64 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*64] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 16; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 4; ++batch_kl) {
        int task_ij0 = blockIdx.x * 256 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 64 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+256];
        double zij = Rp_cache[sq_ij+512];
        double aij = Rp_cache[sq_ij+768];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+64];
        double zkl = Rq_cache[sq_kl+128];
        double akl = Rq_cache[sq_kl+192];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+256];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+512];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_0_1 * dm_ij_cache[sq_ij+0];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl -= R_0_0_0_2 * dm_ij_cache[sq_ij+256];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+512];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+64] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+0];
        double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
        double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
        vj_kl += R_0_0_0_3 * dm_ij_cache[sq_ij+256];
        double R_0_0_1_2 = ypq * R_1_0_0_2;
        vj_kl += R_0_0_1_2 * dm_ij_cache[sq_ij+512];
        double R_0_1_0_2 = xpq * R_1_0_0_2;
        vj_kl += R_0_1_0_2 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+128] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_1_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+256];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl -= R_0_0_2_0 * dm_ij_cache[sq_ij+512];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+192] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_0_1_2 * dm_ij_cache[sq_ij+256];
        double R_1_0_1_1 = ypq * R_2_0_0_1;
        double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_0_2_1 * dm_ij_cache[sq_ij+512];
        double R_0_1_1_1 = xpq * R_1_0_1_1;
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+256] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_0_2_1 * dm_ij_cache[sq_ij+256];
        double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
        double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
        vj_kl += R_0_0_3_0 * dm_ij_cache[sq_ij+512];
        double R_0_1_2_0 = xpq * R_1_0_2_0;
        vj_kl += R_0_1_2_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+320] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_1_0_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+256];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl -= R_0_2_0_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+384] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_1_0_2 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_1 = xpq * R_2_0_0_1;
        double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_2_0_1 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+448] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_1_2_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_1_0 = xpq * R_2_0_1_0;
        double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
        vj_kl += R_0_2_1_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+512] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_2_0_1 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_2_1_0 * dm_ij_cache[sq_ij+512];
        double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
        double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
        vj_kl += R_0_3_0_0 * dm_ij_cache[sq_ij+768];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+576] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+64];
        vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+320];
        vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+384];
        vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+448];
        vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+576];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+64];
        vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+320];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+384];
        vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+448];
        vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+512];
        vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+576];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+64];
        vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+320];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+384];
        vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+448];
        vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+512];
        vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+576];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+64];
        vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+128];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+320];
        vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+384];
        vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+448];
        vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+576];
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
            vj_ij_cache[sq_ij+768] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_ij = blockIdx.x * 256 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*256]);
        }
    }
    for (int n = tx; n < 40; n += 16) {
        int i = n / 4;
        int tile = n % 4;
        int task_kl = blockIdx.y * 64 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*64]);
        }
    }
}

// TILEX=4, TILEY=32, cache_dm=True
__global__
void md_j_2_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 64;
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
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 768;
    double *Rq_cache = Rp_cache + 256;
    double *vj_ij_cache = Rq_cache + 2048;
    double *vj_kl_cache = vj_ij_cache + 640;
    double *vj_cache = vj_kl_cache + 512;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 640;
    // zero out all cache;
    for (int n = sq_id; n < 4864; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 64; n += 256) {
        int task_ij = blockIdx.x * 64 + n;
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
            Rp_cache[n+64] = yij;
            Rp_cache[n+128] = zij;
            Rp_cache[n+192] = aij;
        } else {
            Rp_cache[n+192] = 1.;
        }
    }
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
        } else {
            Rq_cache[n+1536] = 1.;
        }
    }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 4;
        int tile = n % 4;
        int task_ij = blockIdx.x * 64 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*64] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 4; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
        int task_ij0 = blockIdx.x * 64 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+64];
        double zij = Rp_cache[sq_ij+128];
        double aij = Rp_cache[sq_ij+192];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+512];
        double zkl = Rq_cache[sq_kl+1024];
        double akl = Rq_cache[sq_kl+1536];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+64];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+128];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+192];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+256];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+320];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+384];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+448];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+576];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+64] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+128] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+192] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+320] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+384] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+448] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+576] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 4;
        int tile = n % 4;
        int task_ij = blockIdx.x * 64 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*64]);
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=4, TILEY=16, cache_dm=True
__global__
void md_j_2_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 64;
    int task_kl0 = blockIdx.y * 256;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1024;
    double *Rq_cache = Rp_cache + 256;
    double *vj_ij_cache = Rq_cache + 1024;
    double *vj_kl_cache = vj_ij_cache + 640;
    double *vj_cache = vj_kl_cache + 1024;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 640;
    // zero out all cache;
    for (int n = sq_id; n < 4864; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 64; n += 256) {
        int task_ij = blockIdx.x * 64 + n;
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
            Rp_cache[n+64] = yij;
            Rp_cache[n+128] = zij;
            Rp_cache[n+192] = aij;
        } else {
            Rp_cache[n+192] = 1.;
        }
    }
    for (int n = sq_id; n < 256; n += 256) {
        int task_kl = blockIdx.y * 256 + n;
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
            Rq_cache[n+256] = ykl;
            Rq_cache[n+512] = zkl;
            Rq_cache[n+768] = akl;
        } else {
            Rq_cache[n+768] = 1.;
        }
    }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 4;
        int tile = n % 4;
        int task_ij = blockIdx.x * 64 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*64] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*256] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 4; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 16; ++batch_kl) {
        int task_ij0 = blockIdx.x * 64 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 256 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+64];
        double zij = Rp_cache[sq_ij+128];
        double aij = Rp_cache[sq_ij+192];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+256];
        double zkl = Rq_cache[sq_kl+512];
        double akl = Rq_cache[sq_kl+768];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+64];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+128];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+192];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+256];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+320];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+384];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+448];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+576];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_0_1 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_0_2 * dm_ij_cache[sq_ij+64];
        double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
        double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
        vj_kl -= R_0_0_0_3 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+192];
        double R_0_0_1_2 = ypq * R_1_0_0_2;
        vj_kl -= R_0_0_1_2 * dm_ij_cache[sq_ij+256];
        double R_1_0_1_1 = ypq * R_2_0_0_1;
        double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
        vj_kl -= R_0_0_2_1 * dm_ij_cache[sq_ij+320];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+384];
        double R_0_1_0_2 = xpq * R_1_0_0_2;
        vj_kl -= R_0_1_0_2 * dm_ij_cache[sq_ij+448];
        double R_0_1_1_1 = xpq * R_1_0_1_1;
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_1 = xpq * R_2_0_0_1;
        double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
        vj_kl -= R_0_2_0_1 * dm_ij_cache[sq_ij+576];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+256] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_1_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+64];
        vj_kl -= R_0_0_1_2 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_0_2_0 * dm_ij_cache[sq_ij+192];
        vj_kl -= R_0_0_2_1 * dm_ij_cache[sq_ij+256];
        double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
        double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
        vj_kl -= R_0_0_3_0 * dm_ij_cache[sq_ij+320];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+384];
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+448];
        double R_0_1_2_0 = xpq * R_1_0_2_0;
        vj_kl -= R_0_1_2_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_1_0 = xpq * R_2_0_1_0;
        double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
        vj_kl -= R_0_2_1_0 * dm_ij_cache[sq_ij+576];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+512] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_1_0_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+64];
        vj_kl -= R_0_1_0_2 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+192];
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+256];
        vj_kl -= R_0_1_2_0 * dm_ij_cache[sq_ij+320];
        vj_kl -= R_0_2_0_0 * dm_ij_cache[sq_ij+384];
        vj_kl -= R_0_2_0_1 * dm_ij_cache[sq_ij+448];
        vj_kl -= R_0_2_1_0 * dm_ij_cache[sq_ij+512];
        double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
        double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
        vj_kl -= R_0_3_0_0 * dm_ij_cache[sq_ij+576];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+768] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+64] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_3 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+128] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+192] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_3_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+320] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+384] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+448] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_3_0_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+576] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 4;
        int tile = n % 4;
        int task_ij = blockIdx.x * 64 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*64]);
        }
    }
    for (int n = tx; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*256]);
        }
    }
}

// TILEX=8, TILEY=2, cache_dm=True
__global__
void md_j_2_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 128;
    int task_kl0 = blockIdx.y * 32;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1280;
    double *Rq_cache = Rp_cache + 512;
    double *vj_ij_cache = Rq_cache + 128;
    double *vj_kl_cache = vj_ij_cache + 1280;
    double *vj_cache = vj_kl_cache + 320;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 1280;
    // zero out all cache;
    for (int n = sq_id; n < 4096; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 128; n += 256) {
        int task_ij = blockIdx.x * 128 + n;
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
            Rp_cache[n+128] = yij;
            Rp_cache[n+256] = zij;
            Rp_cache[n+384] = aij;
        } else {
            Rp_cache[n+384] = 1.;
        }
    }
    for (int n = sq_id; n < 32; n += 256) {
        int task_kl = blockIdx.y * 32 + n;
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
            Rq_cache[n+32] = ykl;
            Rq_cache[n+64] = zkl;
            Rq_cache[n+96] = akl;
        } else {
            Rq_cache[n+96] = 1.;
        }
    }
    for (int n = ty; n < 80; n += 16) {
        int i = n / 8;
        int tile = n % 8;
        int task_ij = blockIdx.x * 128 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*128] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 20; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_kl = blockIdx.y * 32 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*32] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 8; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 2; ++batch_kl) {
        int task_ij0 = blockIdx.x * 128 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 32 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+128];
        double zij = Rp_cache[sq_ij+256];
        double aij = Rp_cache[sq_ij+384];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+32];
        double zkl = Rq_cache[sq_kl+64];
        double akl = Rq_cache[sq_kl+96];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+128];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+256];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+384];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+512];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+640];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+768];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+896];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+1024];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_0_1 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_0_2 * dm_ij_cache[sq_ij+128];
        double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
        double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
        vj_kl -= R_0_0_0_3 * dm_ij_cache[sq_ij+256];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+384];
        double R_0_0_1_2 = ypq * R_1_0_0_2;
        vj_kl -= R_0_0_1_2 * dm_ij_cache[sq_ij+512];
        double R_1_0_1_1 = ypq * R_2_0_0_1;
        double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
        vj_kl -= R_0_0_2_1 * dm_ij_cache[sq_ij+640];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+768];
        double R_0_1_0_2 = xpq * R_1_0_0_2;
        vj_kl -= R_0_1_0_2 * dm_ij_cache[sq_ij+896];
        double R_0_1_1_1 = xpq * R_1_0_1_1;
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+1024];
        double R_1_1_0_1 = xpq * R_2_0_0_1;
        double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
        vj_kl -= R_0_2_0_1 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+32] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_0_0_3 * dm_ij_cache[sq_ij+128];
        double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
        double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
        double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
        double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
        vj_kl += R_0_0_0_4 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_0_1_2 * dm_ij_cache[sq_ij+384];
        double R_0_0_1_3 = ypq * R_1_0_0_3;
        vj_kl += R_0_0_1_3 * dm_ij_cache[sq_ij+512];
        double R_1_0_1_2 = ypq * R_2_0_0_2;
        double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
        vj_kl += R_0_0_2_2 * dm_ij_cache[sq_ij+640];
        vj_kl += R_0_1_0_2 * dm_ij_cache[sq_ij+768];
        double R_0_1_0_3 = xpq * R_1_0_0_3;
        vj_kl += R_0_1_0_3 * dm_ij_cache[sq_ij+896];
        double R_0_1_1_2 = xpq * R_1_0_1_2;
        vj_kl += R_0_1_1_2 * dm_ij_cache[sq_ij+1024];
        double R_1_1_0_2 = xpq * R_2_0_0_2;
        double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
        vj_kl += R_0_2_0_2 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+64] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_1_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_0_1_2 * dm_ij_cache[sq_ij+256];
        vj_kl -= R_0_0_2_0 * dm_ij_cache[sq_ij+384];
        vj_kl -= R_0_0_2_1 * dm_ij_cache[sq_ij+512];
        double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
        double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
        vj_kl -= R_0_0_3_0 * dm_ij_cache[sq_ij+640];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+768];
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+896];
        double R_0_1_2_0 = xpq * R_1_0_2_0;
        vj_kl -= R_0_1_2_0 * dm_ij_cache[sq_ij+1024];
        double R_1_1_1_0 = xpq * R_2_0_1_0;
        double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
        vj_kl -= R_0_2_1_0 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+96] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_0_1_2 * dm_ij_cache[sq_ij+128];
        vj_kl += R_0_0_1_3 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_0_2_1 * dm_ij_cache[sq_ij+384];
        vj_kl += R_0_0_2_2 * dm_ij_cache[sq_ij+512];
        double R_2_0_1_1 = ypq * R_3_0_0_1;
        double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
        double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
        vj_kl += R_0_0_3_1 * dm_ij_cache[sq_ij+640];
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+768];
        vj_kl += R_0_1_1_2 * dm_ij_cache[sq_ij+896];
        double R_0_1_2_1 = xpq * R_1_0_2_1;
        vj_kl += R_0_1_2_1 * dm_ij_cache[sq_ij+1024];
        double R_1_1_1_1 = xpq * R_2_0_1_1;
        double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
        vj_kl += R_0_2_1_1 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+128] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_0_2_1 * dm_ij_cache[sq_ij+128];
        vj_kl += R_0_0_2_2 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_0_3_0 * dm_ij_cache[sq_ij+384];
        vj_kl += R_0_0_3_1 * dm_ij_cache[sq_ij+512];
        double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
        double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
        double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
        double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
        vj_kl += R_0_0_4_0 * dm_ij_cache[sq_ij+640];
        vj_kl += R_0_1_2_0 * dm_ij_cache[sq_ij+768];
        vj_kl += R_0_1_2_1 * dm_ij_cache[sq_ij+896];
        double R_0_1_3_0 = xpq * R_1_0_3_0;
        vj_kl += R_0_1_3_0 * dm_ij_cache[sq_ij+1024];
        double R_1_1_2_0 = xpq * R_2_0_2_0;
        double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
        vj_kl += R_0_2_2_0 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+160] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_1_0_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_1_0_2 * dm_ij_cache[sq_ij+256];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+384];
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+512];
        vj_kl -= R_0_1_2_0 * dm_ij_cache[sq_ij+640];
        vj_kl -= R_0_2_0_0 * dm_ij_cache[sq_ij+768];
        vj_kl -= R_0_2_0_1 * dm_ij_cache[sq_ij+896];
        vj_kl -= R_0_2_1_0 * dm_ij_cache[sq_ij+1024];
        double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
        double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
        vj_kl -= R_0_3_0_0 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+192] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_1_0_2 * dm_ij_cache[sq_ij+128];
        vj_kl += R_0_1_0_3 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+384];
        vj_kl += R_0_1_1_2 * dm_ij_cache[sq_ij+512];
        vj_kl += R_0_1_2_1 * dm_ij_cache[sq_ij+640];
        vj_kl += R_0_2_0_1 * dm_ij_cache[sq_ij+768];
        vj_kl += R_0_2_0_2 * dm_ij_cache[sq_ij+896];
        vj_kl += R_0_2_1_1 * dm_ij_cache[sq_ij+1024];
        double R_2_1_0_1 = xpq * R_3_0_0_1;
        double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
        double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
        vj_kl += R_0_3_0_1 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+224] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+128];
        vj_kl += R_0_1_1_2 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_1_2_0 * dm_ij_cache[sq_ij+384];
        vj_kl += R_0_1_2_1 * dm_ij_cache[sq_ij+512];
        vj_kl += R_0_1_3_0 * dm_ij_cache[sq_ij+640];
        vj_kl += R_0_2_1_0 * dm_ij_cache[sq_ij+768];
        vj_kl += R_0_2_1_1 * dm_ij_cache[sq_ij+896];
        vj_kl += R_0_2_2_0 * dm_ij_cache[sq_ij+1024];
        double R_2_1_1_0 = xpq * R_3_0_1_0;
        double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
        double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
        vj_kl += R_0_3_1_0 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+256] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+0];
        vj_kl += R_0_2_0_1 * dm_ij_cache[sq_ij+128];
        vj_kl += R_0_2_0_2 * dm_ij_cache[sq_ij+256];
        vj_kl += R_0_2_1_0 * dm_ij_cache[sq_ij+384];
        vj_kl += R_0_2_1_1 * dm_ij_cache[sq_ij+512];
        vj_kl += R_0_2_2_0 * dm_ij_cache[sq_ij+640];
        vj_kl += R_0_3_0_0 * dm_ij_cache[sq_ij+768];
        vj_kl += R_0_3_0_1 * dm_ij_cache[sq_ij+896];
        vj_kl += R_0_3_1_0 * dm_ij_cache[sq_ij+1024];
        double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
        double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
        double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
        double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
        vj_kl += R_0_4_0_0 * dm_ij_cache[sq_ij+1152];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+288] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+128] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_3 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_0_0_4 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_0_1_3 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_0_2_2 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_1_0_3 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_2_0_2 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+384] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_0_1_3 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_0_2_2 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_0_3_1 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_0_2_2 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_0_3_0 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_0_3_1 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_0_4_0 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_1_3_0 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_2_2_0 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+640] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+768] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_1_0_3 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_2_0_2 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_3_0_1 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+896] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_1_1_2 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_1_2_1 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_1_3_0 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_2_2_0 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_3_1_0 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+1024] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+32];
        vj_ij += R_0_2_0_2 * dm_kl_cache[sq_kl+64];
        vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+96];
        vj_ij += R_0_2_1_1 * dm_kl_cache[sq_kl+128];
        vj_ij += R_0_2_2_0 * dm_kl_cache[sq_kl+160];
        vj_ij -= R_0_3_0_0 * dm_kl_cache[sq_kl+192];
        vj_ij += R_0_3_0_1 * dm_kl_cache[sq_kl+224];
        vj_ij += R_0_3_1_0 * dm_kl_cache[sq_kl+256];
        vj_ij += R_0_4_0_0 * dm_kl_cache[sq_kl+288];
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
            vj_ij_cache[sq_ij+1152] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 80; n += 16) {
        int i = n / 8;
        int tile = n % 8;
        int task_ij = blockIdx.x * 128 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*128]);
        }
    }
    for (int n = tx; n < 20; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_kl = blockIdx.y * 32 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*32]);
        }
    }
}

// TILEX=2, TILEY=32, cache_dm=True
__global__
void md_j_3_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 32;
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
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1024;
    double *Rq_cache = Rp_cache + 128;
    double *vj_ij_cache = Rq_cache + 2048;
    double *vj_kl_cache = vj_ij_cache + 640;
    double *vj_cache = vj_kl_cache + 512;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 640;
    // zero out all cache;
    for (int n = sq_id; n < 4736; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 32; n += 256) {
        int task_ij = blockIdx.x * 32 + n;
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
            Rp_cache[n+32] = yij;
            Rp_cache[n+64] = zij;
            Rp_cache[n+96] = aij;
        } else {
            Rp_cache[n+96] = 1.;
        }
    }
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
        } else {
            Rq_cache[n+1536] = 1.;
        }
    }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_ij = blockIdx.x * 32 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*32] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*512] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 2; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 32; ++batch_kl) {
        int task_ij0 = blockIdx.x * 32 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 512 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+32];
        double zij = Rp_cache[sq_ij+64];
        double aij = Rp_cache[sq_ij+96];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+512];
        double zkl = Rq_cache[sq_kl+1024];
        double akl = Rq_cache[sq_kl+1536];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+32];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+64];
        double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
        double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
        vj_kl += R_0_0_0_3 * dm_ij_cache[sq_ij+96];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+128];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+160];
        double R_0_0_1_2 = ypq * R_1_0_0_2;
        vj_kl += R_0_0_1_2 * dm_ij_cache[sq_ij+192];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+224];
        double R_1_0_1_1 = ypq * R_2_0_0_1;
        double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_0_2_1 * dm_ij_cache[sq_ij+256];
        double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
        double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
        vj_kl += R_0_0_3_0 * dm_ij_cache[sq_ij+288];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+320];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+352];
        double R_0_1_0_2 = xpq * R_1_0_0_2;
        vj_kl += R_0_1_0_2 * dm_ij_cache[sq_ij+384];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+416];
        double R_0_1_1_1 = xpq * R_1_0_1_1;
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+448];
        double R_0_1_2_0 = xpq * R_1_0_2_0;
        vj_kl += R_0_1_2_0 * dm_ij_cache[sq_ij+480];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_1 = xpq * R_2_0_0_1;
        double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_2_0_1 * dm_ij_cache[sq_ij+544];
        double R_1_1_1_0 = xpq * R_2_0_1_0;
        double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
        vj_kl += R_0_2_1_0 * dm_ij_cache[sq_ij+576];
        double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
        double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
        vj_kl += R_0_3_0_0 * dm_ij_cache[sq_ij+608];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+32] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+64] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+96] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+128] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+160] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+192] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+224] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+288] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+320] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+352] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+384] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+416] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+448] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+480] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+544] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+576] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+608] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_ij = blockIdx.x * 32 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*32]);
        }
    }
    for (int n = tx; n < 32; n += 16) {
        int i = n / 32;
        int tile = n % 32;
        int task_kl = blockIdx.y * 512 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*512]);
        }
    }
}

// TILEX=2, TILEY=16, cache_dm=True
__global__
void md_j_3_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 32;
    int task_kl0 = blockIdx.y * 256;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1280;
    double *Rq_cache = Rp_cache + 128;
    double *vj_ij_cache = Rq_cache + 1024;
    double *vj_kl_cache = vj_ij_cache + 640;
    double *vj_cache = vj_kl_cache + 1024;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 640;
    // zero out all cache;
    for (int n = sq_id; n < 4736; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 32; n += 256) {
        int task_ij = blockIdx.x * 32 + n;
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
            Rp_cache[n+32] = yij;
            Rp_cache[n+64] = zij;
            Rp_cache[n+96] = aij;
        } else {
            Rp_cache[n+96] = 1.;
        }
    }
    for (int n = sq_id; n < 256; n += 256) {
        int task_kl = blockIdx.y * 256 + n;
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
            Rq_cache[n+256] = ykl;
            Rq_cache[n+512] = zkl;
            Rq_cache[n+768] = akl;
        } else {
            Rq_cache[n+768] = 1.;
        }
    }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_ij = blockIdx.x * 32 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*32] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*256] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 2; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 16; ++batch_kl) {
        int task_ij0 = blockIdx.x * 32 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 256 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+32];
        double zij = Rp_cache[sq_ij+64];
        double aij = Rp_cache[sq_ij+96];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+256];
        double zkl = Rq_cache[sq_kl+512];
        double akl = Rq_cache[sq_kl+768];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+32];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+64];
        double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
        double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
        vj_kl += R_0_0_0_3 * dm_ij_cache[sq_ij+96];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+128];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+160];
        double R_0_0_1_2 = ypq * R_1_0_0_2;
        vj_kl += R_0_0_1_2 * dm_ij_cache[sq_ij+192];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+224];
        double R_1_0_1_1 = ypq * R_2_0_0_1;
        double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_0_2_1 * dm_ij_cache[sq_ij+256];
        double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
        double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
        vj_kl += R_0_0_3_0 * dm_ij_cache[sq_ij+288];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+320];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+352];
        double R_0_1_0_2 = xpq * R_1_0_0_2;
        vj_kl += R_0_1_0_2 * dm_ij_cache[sq_ij+384];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+416];
        double R_0_1_1_1 = xpq * R_1_0_1_1;
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+448];
        double R_0_1_2_0 = xpq * R_1_0_2_0;
        vj_kl += R_0_1_2_0 * dm_ij_cache[sq_ij+480];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_1 = xpq * R_2_0_0_1;
        double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_2_0_1 * dm_ij_cache[sq_ij+544];
        double R_1_1_1_0 = xpq * R_2_0_1_0;
        double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
        vj_kl += R_0_2_1_0 * dm_ij_cache[sq_ij+576];
        double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
        double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
        vj_kl += R_0_3_0_0 * dm_ij_cache[sq_ij+608];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_0_1 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_0_2 * dm_ij_cache[sq_ij+32];
        vj_kl -= R_0_0_0_3 * dm_ij_cache[sq_ij+64];
        double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
        double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
        double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
        double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
        vj_kl -= R_0_0_0_4 * dm_ij_cache[sq_ij+96];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_0_1_2 * dm_ij_cache[sq_ij+160];
        double R_0_0_1_3 = ypq * R_1_0_0_3;
        vj_kl -= R_0_0_1_3 * dm_ij_cache[sq_ij+192];
        vj_kl -= R_0_0_2_1 * dm_ij_cache[sq_ij+224];
        double R_1_0_1_2 = ypq * R_2_0_0_2;
        double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
        vj_kl -= R_0_0_2_2 * dm_ij_cache[sq_ij+256];
        double R_2_0_1_1 = ypq * R_3_0_0_1;
        double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
        double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
        vj_kl -= R_0_0_3_1 * dm_ij_cache[sq_ij+288];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+320];
        vj_kl -= R_0_1_0_2 * dm_ij_cache[sq_ij+352];
        double R_0_1_0_3 = xpq * R_1_0_0_3;
        vj_kl -= R_0_1_0_3 * dm_ij_cache[sq_ij+384];
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+416];
        double R_0_1_1_2 = xpq * R_1_0_1_2;
        vj_kl -= R_0_1_1_2 * dm_ij_cache[sq_ij+448];
        double R_0_1_2_1 = xpq * R_1_0_2_1;
        vj_kl -= R_0_1_2_1 * dm_ij_cache[sq_ij+480];
        vj_kl -= R_0_2_0_1 * dm_ij_cache[sq_ij+512];
        double R_1_1_0_2 = xpq * R_2_0_0_2;
        double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
        vj_kl -= R_0_2_0_2 * dm_ij_cache[sq_ij+544];
        double R_1_1_1_1 = xpq * R_2_0_1_1;
        double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
        vj_kl -= R_0_2_1_1 * dm_ij_cache[sq_ij+576];
        double R_2_1_0_1 = xpq * R_3_0_0_1;
        double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
        double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
        vj_kl -= R_0_3_0_1 * dm_ij_cache[sq_ij+608];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+256] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_0_1_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_0_1_1 * dm_ij_cache[sq_ij+32];
        vj_kl -= R_0_0_1_2 * dm_ij_cache[sq_ij+64];
        vj_kl -= R_0_0_1_3 * dm_ij_cache[sq_ij+96];
        vj_kl -= R_0_0_2_0 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_0_2_1 * dm_ij_cache[sq_ij+160];
        vj_kl -= R_0_0_2_2 * dm_ij_cache[sq_ij+192];
        vj_kl -= R_0_0_3_0 * dm_ij_cache[sq_ij+224];
        vj_kl -= R_0_0_3_1 * dm_ij_cache[sq_ij+256];
        double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
        double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
        double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
        double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
        vj_kl -= R_0_0_4_0 * dm_ij_cache[sq_ij+288];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+320];
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+352];
        vj_kl -= R_0_1_1_2 * dm_ij_cache[sq_ij+384];
        vj_kl -= R_0_1_2_0 * dm_ij_cache[sq_ij+416];
        vj_kl -= R_0_1_2_1 * dm_ij_cache[sq_ij+448];
        double R_0_1_3_0 = xpq * R_1_0_3_0;
        vj_kl -= R_0_1_3_0 * dm_ij_cache[sq_ij+480];
        vj_kl -= R_0_2_1_0 * dm_ij_cache[sq_ij+512];
        vj_kl -= R_0_2_1_1 * dm_ij_cache[sq_ij+544];
        double R_1_1_2_0 = xpq * R_2_0_2_0;
        double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
        vj_kl -= R_0_2_2_0 * dm_ij_cache[sq_ij+576];
        double R_2_1_1_0 = xpq * R_3_0_1_0;
        double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
        double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
        vj_kl -= R_0_3_1_0 * dm_ij_cache[sq_ij+608];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+512] += vj_cache[sq_id];
        }
        vj_kl = 0.;
        vj_kl -= R_0_1_0_0 * dm_ij_cache[sq_ij+0];
        vj_kl -= R_0_1_0_1 * dm_ij_cache[sq_ij+32];
        vj_kl -= R_0_1_0_2 * dm_ij_cache[sq_ij+64];
        vj_kl -= R_0_1_0_3 * dm_ij_cache[sq_ij+96];
        vj_kl -= R_0_1_1_0 * dm_ij_cache[sq_ij+128];
        vj_kl -= R_0_1_1_1 * dm_ij_cache[sq_ij+160];
        vj_kl -= R_0_1_1_2 * dm_ij_cache[sq_ij+192];
        vj_kl -= R_0_1_2_0 * dm_ij_cache[sq_ij+224];
        vj_kl -= R_0_1_2_1 * dm_ij_cache[sq_ij+256];
        vj_kl -= R_0_1_3_0 * dm_ij_cache[sq_ij+288];
        vj_kl -= R_0_2_0_0 * dm_ij_cache[sq_ij+320];
        vj_kl -= R_0_2_0_1 * dm_ij_cache[sq_ij+352];
        vj_kl -= R_0_2_0_2 * dm_ij_cache[sq_ij+384];
        vj_kl -= R_0_2_1_0 * dm_ij_cache[sq_ij+416];
        vj_kl -= R_0_2_1_1 * dm_ij_cache[sq_ij+448];
        vj_kl -= R_0_2_2_0 * dm_ij_cache[sq_ij+480];
        vj_kl -= R_0_3_0_0 * dm_ij_cache[sq_ij+512];
        vj_kl -= R_0_3_0_1 * dm_ij_cache[sq_ij+544];
        vj_kl -= R_0_3_1_0 * dm_ij_cache[sq_ij+576];
        double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
        double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
        double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
        double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
        vj_kl -= R_0_4_0_0 * dm_ij_cache[sq_ij+608];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+768] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += gamma_inc[sq_id+0*256] * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_0_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+32] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_2 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_3 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+64] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_0_3 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_0_4 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_1_3 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_0_3 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+96] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_2_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+128] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+160] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_1_2 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_1_3 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_2_2 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_1_2 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+192] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_2_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_2_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_3_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+224] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_2_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_2_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_3_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_2_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_0_3_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_0_3_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_0_4_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_1_3_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+288] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_0_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+320] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+352] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_0_2 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_0_3 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_1_2 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_0_2 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+384] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_2_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+416] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_1_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_1_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_2_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_1_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+448] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_1_2_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_1_2_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_1_3_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_2_2_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+480] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_2_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_2_0_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_2_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_3_0_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_2_0_1 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_2_0_2 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_2_1_1 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_3_0_1 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+544] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_2_1_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_2_1_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_2_2_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_3_1_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+576] += vj_cache[sq_id];
        }
        vj_ij = 0.;
        vj_ij += R_0_3_0_0 * dm_kl_cache[sq_kl+0];
        vj_ij -= R_0_3_0_1 * dm_kl_cache[sq_kl+256];
        vj_ij -= R_0_3_1_0 * dm_kl_cache[sq_kl+512];
        vj_ij -= R_0_4_0_0 * dm_kl_cache[sq_kl+768];
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
            vj_ij_cache[sq_ij+608] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 40; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_ij = blockIdx.x * 32 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*32]);
        }
    }
    for (int n = tx; n < 64; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*256]);
        }
    }
}

// TILEX=2, TILEY=16, cache_dm=True
__global__
void md_j_4_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.tile_ij_mapping;
    int *pair_kl_mapping = bounds.tile_kl_mapping;
    int task_ij0 = blockIdx.x * 32;
    int task_kl0 = blockIdx.y * 256;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int *bas = envs.bas;
    int *dm_pair_loc = envs.ao_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_ij, vj_kl;

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double gamma_inc[];
    double *Rp_cache = gamma_inc + 1280;
    double *Rq_cache = Rp_cache + 128;
    double *vj_ij_cache = Rq_cache + 1024;
    double *vj_kl_cache = vj_ij_cache + 1120;
    double *vj_cache = vj_kl_cache + 256;
    double *dm_ij_cache = vj_cache + 256;
    double *dm_kl_cache = dm_ij_cache + 1120;
    // zero out all cache;
    for (int n = sq_id; n < 4160; n += 256) {
        Rp_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = sq_id; n < 32; n += 256) {
        int task_ij = blockIdx.x * 32 + n;
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
            Rp_cache[n+32] = yij;
            Rp_cache[n+64] = zij;
            Rp_cache[n+96] = aij;
        } else {
            Rp_cache[n+96] = 1.;
        }
    }
    for (int n = sq_id; n < 256; n += 256) {
        int task_kl = blockIdx.y * 256 + n;
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
            Rq_cache[n+256] = ykl;
            Rq_cache[n+512] = zkl;
            Rq_cache[n+768] = akl;
        } else {
            Rq_cache[n+768] = 1.;
        }
    }
    for (int n = ty; n < 70; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_ij = blockIdx.x * 32 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            dm_ij_cache[sq_ij+i*32] = dm[dm_ij_pair0+i];
        }
    }
    for (int n = tx; n < 16; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            dm_kl_cache[sq_kl+i*256] = dm[dm_kl_pair0+i];
        }
    }
    __syncthreads();

    for (int batch_ij = 0; batch_ij < 2; ++batch_ij) {
    for (int batch_kl = 0; batch_kl < 16; ++batch_kl) {
        int task_ij0 = blockIdx.x * 32 + batch_ij * 16;
        int task_kl0 = blockIdx.y * 256 + batch_kl * 16;
        if (task_ij0 >= npairs_ij || task_kl0 >= npairs_kl) {
            continue;
        }
        int pair_ij0 = pair_ij_mapping[task_ij0];
        int pair_kl0 = pair_kl_mapping[task_kl0];
        if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
            continue;
        }

        int sq_ij = tx + batch_ij * 16;
        int sq_kl = ty + batch_kl * 16;
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
            if (task_ij < task_kl) fac_sym = 0.;
        }
        double xij = Rp_cache[sq_ij+0];
        double yij = Rp_cache[sq_ij+32];
        double zij = Rp_cache[sq_ij+64];
        double aij = Rp_cache[sq_ij+96];
        double xkl = Rq_cache[sq_kl+0];
        double ykl = Rq_cache[sq_kl+256];
        double zkl = Rq_cache[sq_kl+512];
        double akl = Rq_cache[sq_kl+768];
        double fac = fac_sym / (aij*akl*sqrt(aij+akl));
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
        vj_kl += gamma_inc[sq_id+0*256] * dm_ij_cache[sq_ij+0];
        double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_1 * dm_ij_cache[sq_ij+32];
        double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
        double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_0_2 * dm_ij_cache[sq_ij+64];
        double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
        double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
        vj_kl += R_0_0_0_3 * dm_ij_cache[sq_ij+96];
        double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
        double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
        double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
        double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
        vj_kl += R_0_0_0_4 * dm_ij_cache[sq_ij+128];
        double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_1_0 * dm_ij_cache[sq_ij+160];
        double R_0_0_1_1 = ypq * R_1_0_0_1;
        vj_kl += R_0_0_1_1 * dm_ij_cache[sq_ij+192];
        double R_0_0_1_2 = ypq * R_1_0_0_2;
        vj_kl += R_0_0_1_2 * dm_ij_cache[sq_ij+224];
        double R_0_0_1_3 = ypq * R_1_0_0_3;
        vj_kl += R_0_0_1_3 * dm_ij_cache[sq_ij+256];
        double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
        double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_0_2_0 * dm_ij_cache[sq_ij+288];
        double R_1_0_1_1 = ypq * R_2_0_0_1;
        double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_0_2_1 * dm_ij_cache[sq_ij+320];
        double R_1_0_1_2 = ypq * R_2_0_0_2;
        double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
        vj_kl += R_0_0_2_2 * dm_ij_cache[sq_ij+352];
        double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
        double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
        vj_kl += R_0_0_3_0 * dm_ij_cache[sq_ij+384];
        double R_2_0_1_1 = ypq * R_3_0_0_1;
        double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
        double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
        vj_kl += R_0_0_3_1 * dm_ij_cache[sq_ij+416];
        double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
        double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
        double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
        double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
        vj_kl += R_0_0_4_0 * dm_ij_cache[sq_ij+448];
        double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
        vj_kl += R_0_1_0_0 * dm_ij_cache[sq_ij+480];
        double R_0_1_0_1 = xpq * R_1_0_0_1;
        vj_kl += R_0_1_0_1 * dm_ij_cache[sq_ij+512];
        double R_0_1_0_2 = xpq * R_1_0_0_2;
        vj_kl += R_0_1_0_2 * dm_ij_cache[sq_ij+544];
        double R_0_1_0_3 = xpq * R_1_0_0_3;
        vj_kl += R_0_1_0_3 * dm_ij_cache[sq_ij+576];
        double R_0_1_1_0 = xpq * R_1_0_1_0;
        vj_kl += R_0_1_1_0 * dm_ij_cache[sq_ij+608];
        double R_0_1_1_1 = xpq * R_1_0_1_1;
        vj_kl += R_0_1_1_1 * dm_ij_cache[sq_ij+640];
        double R_0_1_1_2 = xpq * R_1_0_1_2;
        vj_kl += R_0_1_1_2 * dm_ij_cache[sq_ij+672];
        double R_0_1_2_0 = xpq * R_1_0_2_0;
        vj_kl += R_0_1_2_0 * dm_ij_cache[sq_ij+704];
        double R_0_1_2_1 = xpq * R_1_0_2_1;
        vj_kl += R_0_1_2_1 * dm_ij_cache[sq_ij+736];
        double R_0_1_3_0 = xpq * R_1_0_3_0;
        vj_kl += R_0_1_3_0 * dm_ij_cache[sq_ij+768];
        double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
        double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
        vj_kl += R_0_2_0_0 * dm_ij_cache[sq_ij+800];
        double R_1_1_0_1 = xpq * R_2_0_0_1;
        double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
        vj_kl += R_0_2_0_1 * dm_ij_cache[sq_ij+832];
        double R_1_1_0_2 = xpq * R_2_0_0_2;
        double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
        vj_kl += R_0_2_0_2 * dm_ij_cache[sq_ij+864];
        double R_1_1_1_0 = xpq * R_2_0_1_0;
        double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
        vj_kl += R_0_2_1_0 * dm_ij_cache[sq_ij+896];
        double R_1_1_1_1 = xpq * R_2_0_1_1;
        double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
        vj_kl += R_0_2_1_1 * dm_ij_cache[sq_ij+928];
        double R_1_1_2_0 = xpq * R_2_0_2_0;
        double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
        vj_kl += R_0_2_2_0 * dm_ij_cache[sq_ij+960];
        double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
        double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
        double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
        vj_kl += R_0_3_0_0 * dm_ij_cache[sq_ij+992];
        double R_2_1_0_1 = xpq * R_3_0_0_1;
        double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
        double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
        vj_kl += R_0_3_0_1 * dm_ij_cache[sq_ij+1024];
        double R_2_1_1_0 = xpq * R_3_0_1_0;
        double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
        double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
        vj_kl += R_0_3_1_0 * dm_ij_cache[sq_ij+1056];
        double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
        double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
        double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
        double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
        vj_kl += R_0_4_0_0 * dm_ij_cache[sq_ij+1088];
        __syncthreads();
        vj_cache[sq_id] = vj_kl;
        for (int stride = 8; stride > 0; stride /= 2) {
            __syncthreads();
            if (tx < stride) {
                vj_cache[sq_id] += vj_cache[sq_id + stride];
            }
        }
        __syncthreads();
        if (tx == 0 && task_kl0+ty < npairs_kl) {
            vj_kl_cache[sq_kl+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+0] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+32] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+64] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+96] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+128] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+160] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+192] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+224] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+256] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+288] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+320] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+352] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+384] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+416] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+448] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+480] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+512] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+544] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+576] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+608] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+640] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+672] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+704] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+736] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+768] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+800] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+832] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+864] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+896] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+928] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+960] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+992] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+1024] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+1056] += vj_cache[sq_id];
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
            vj_ij_cache[sq_ij+1088] += vj_cache[sq_id];
        }
        __syncthreads();
    } }
    for (int n = ty; n < 70; n += 16) {
        int i = n / 2;
        int tile = n % 2;
        int task_ij = blockIdx.x * 32 + tile * 16 + tx;
        if (task_ij < npairs_ij) {
            int pair_ij = pair_ij_mapping[task_ij];
            int dm_ij_pair0 = dm_pair_loc[pair_ij];
            int sq_ij = tx + tile * 16;
            atomicAdd(vj+dm_ij_pair0+i, vj_ij_cache[sq_ij+i*32]);
        }
    }
    for (int n = tx; n < 16; n += 16) {
        int i = n / 16;
        int tile = n % 16;
        int task_kl = blockIdx.y * 256 + tile * 16 + ty;
        if (task_kl < npairs_kl) {
            int pair_kl = pair_kl_mapping[task_kl];
            int dm_kl_pair0 = dm_pair_loc[pair_kl];
            int sq_kl = ty + tile * 16;
            atomicAdd(vj+dm_kl_pair0+i, vj_kl_cache[sq_kl+i*256]);
        }
    }
}

int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds, int workers)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int lij = li + lj;
    int lkl = lk + ll;
    dim3 threads(16, 16);
    dim3 blocks;
    int ijkl = lij*9 + lkl;
    switch (ijkl) {
    case 0: // lij=0, lkl=0, tilex=32, tiley=16
        blocks.x = (bounds->npairs_ij + 511) / 512;
        blocks.y = (bounds->npairs_kl + 255) / 256;
        md_j_0_0<<<blocks, threads, 5120*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 9: // lij=1, lkl=0, tilex=8, tiley=32
        blocks.x = (bounds->npairs_ij + 127) / 128;
        blocks.y = (bounds->npairs_kl + 511) / 512;
        md_j_1_0<<<blocks, threads, 5376*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 10: // lij=1, lkl=1, tilex=16, tiley=8
        blocks.x = (bounds->npairs_ij + 255) / 256;
        blocks.y = (bounds->npairs_kl + 127) / 128;
        md_j_1_1<<<blocks, threads, 5632*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 11: // lij=1, lkl=2, tilex=16, tiley=4
        blocks.x = (bounds->npairs_ij + 255) / 256;
        blocks.y = (bounds->npairs_kl + 63) / 64;
        md_j_1_2<<<blocks, threads, 5888*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 18: // lij=2, lkl=0, tilex=4, tiley=32
        blocks.x = (bounds->npairs_ij + 63) / 64;
        blocks.y = (bounds->npairs_kl + 511) / 512;
        md_j_2_0<<<blocks, threads, 5632*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 19: // lij=2, lkl=1, tilex=4, tiley=16
        blocks.x = (bounds->npairs_ij + 63) / 64;
        blocks.y = (bounds->npairs_kl + 255) / 256;
        md_j_2_1<<<blocks, threads, 5888*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 20: // lij=2, lkl=2, tilex=8, tiley=2
        blocks.x = (bounds->npairs_ij + 127) / 128;
        blocks.y = (bounds->npairs_kl + 31) / 32;
        md_j_2_2<<<blocks, threads, 5376*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 27: // lij=3, lkl=0, tilex=2, tiley=32
        blocks.x = (bounds->npairs_ij + 31) / 32;
        blocks.y = (bounds->npairs_kl + 511) / 512;
        md_j_3_0<<<blocks, threads, 5760*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 28: // lij=3, lkl=1, tilex=2, tiley=16
        blocks.x = (bounds->npairs_ij + 31) / 32;
        blocks.y = (bounds->npairs_kl + 255) / 256;
        md_j_3_1<<<blocks, threads, 6016*sizeof(double)>>>(*envs, *jk, *bounds); break;
    case 36: // lij=4, lkl=0, tilex=2, tiley=16
        blocks.x = (bounds->npairs_ij + 31) / 32;
        blocks.y = (bounds->npairs_kl + 255) / 256;
        md_j_4_0<<<blocks, threads, 5440*sizeof(double)>>>(*envs, *jk, *bounds); break;
    default: return 0;
    }
    return 1;
}
