#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/gamma_inc.cu"
#include "gvhf-md/md_j.cuh"


// TILEX=34, TILEY=34
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_0_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 544;
    int task_kl0 = blockIdx.y * 544;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+544 <= task_kl0) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 544;
    double *Rp_cache = Rq_cache + 2176;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 16;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2784; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 544; n += 256) {
        int task_kl = blockIdx.y * 544 + n;
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
            Rq_cache[n+544] = ykl;
            Rq_cache[n+1088] = zkl;
            Rq_cache[n+1632] = akl;
        } else {
            Rq_cache[n+1632] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 34; ++batch_ij) {
        int task_ij0 = blockIdx.x * 544 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 1; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[1];
        for (int ij = 0; ij < 1; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 34; ++batch_kl) {
            int task_kl0 = blockIdx.y * 544 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*34] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*34] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
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
            double ykl = Rq_cache[sq_kl+544];
            double zkl = Rq_cache[sq_kl+1088];
            double akl = Rq_cache[sq_kl+1632];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 0, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 0; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 1; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 34; n += 16) {
        int batch_kl = n % 34;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 544 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 34;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*544]);
        }
    }
}

// TILEX=48, TILEY=30
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_1_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 480;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 480;
    double *Rp_cache = Rq_cache + 1920;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2464; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 480; n += 256) {
        int task_kl = blockIdx.y * 480 + n;
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
            Rq_cache[n+480] = ykl;
            Rq_cache[n+960] = zkl;
            Rq_cache[n+1440] = akl;
        } else {
            Rq_cache[n+1440] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[4];
        for (int ij = 0; ij < 4; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 30; ++batch_kl) {
            int task_kl0 = blockIdx.y * 480 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*30] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+480];
            double zkl = Rq_cache[sq_kl+960];
            double akl = Rq_cache[sq_kl+1440];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 1, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 1; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+32];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_1_0 * dm_kl0;
            vj_ij[3] += R_0_1_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 30; n += 16) {
        int batch_kl = n % 30;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 480 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 30;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*480]);
        }
    }
}

// TILEX=17, TILEY=17
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_1_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 272;
    int task_kl0 = blockIdx.y * 272;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+272 <= task_kl0) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1088;
    double *Rp_cache = Rq_cache + 1088;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2240; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 272; n += 256) {
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
        } else {
            Rq_cache[n+816] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 17; ++batch_ij) {
        int task_ij0 = blockIdx.x * 272 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[4];
        for (int ij = 0; ij < 4; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 17; ++batch_kl) {
            int task_kl0 = blockIdx.y * 272 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*17] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*17] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
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
            eval_gamma_inc_fn(gamma_inc, theta_rr, 2, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 2; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+32];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[tx+0];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+32];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+272] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+16];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[tx+32];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+544] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+32];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+816] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_1_0 * dm_kl0;
            vj_ij[3] += R_0_1_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+1];
            vj_ij[0] -= R_0_0_0_1 * dm_kl0;
            vj_ij[1] -= R_0_0_0_2 * dm_kl0;
            vj_ij[2] -= R_0_0_1_1 * dm_kl0;
            vj_ij[3] -= R_0_1_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+2];
            vj_ij[0] -= R_0_0_1_0 * dm_kl0;
            vj_ij[1] -= R_0_0_1_1 * dm_kl0;
            vj_ij[2] -= R_0_0_2_0 * dm_kl0;
            vj_ij[3] -= R_0_1_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+3];
            vj_ij[0] -= R_0_1_0_0 * dm_kl0;
            vj_ij[1] -= R_0_1_0_1 * dm_kl0;
            vj_ij[2] -= R_0_1_1_0 * dm_kl0;
            vj_ij[3] -= R_0_2_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 68; n += 16) {
        int batch_kl = n % 17;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 272 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 17;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*272]);
        }
    }
}

// TILEX=48, TILEY=26
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_2_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 416;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 416;
    double *Rp_cache = Rq_cache + 1664;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2144; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 416; n += 256) {
        int task_kl = blockIdx.y * 416 + n;
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
            Rq_cache[n+416] = ykl;
            Rq_cache[n+832] = zkl;
            Rq_cache[n+1248] = akl;
        } else {
            Rq_cache[n+1248] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[10];
        for (int ij = 0; ij < 10; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 26; ++batch_kl) {
            int task_kl0 = blockIdx.y * 416 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*26] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+416];
            double zkl = Rq_cache[sq_kl+832];
            double akl = Rq_cache[sq_kl+1248];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 2, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 2; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_1_0 * dm_kl0;
            vj_ij[4] += R_0_0_1_1 * dm_kl0;
            vj_ij[5] += R_0_0_2_0 * dm_kl0;
            vj_ij[6] += R_0_1_0_0 * dm_kl0;
            vj_ij[7] += R_0_1_0_1 * dm_kl0;
            vj_ij[8] += R_0_1_1_0 * dm_kl0;
            vj_ij[9] += R_0_2_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 10; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 26; n += 16) {
        int batch_kl = n % 26;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 416 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 26;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*416]);
        }
    }
}

// TILEX=48, TILEY=14
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_2_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 896;
    double *Rp_cache = Rq_cache + 896;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 1856; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 224; n += 256) {
        int task_kl = blockIdx.y * 224 + n;
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
            Rq_cache[n+224] = ykl;
            Rq_cache[n+448] = zkl;
            Rq_cache[n+672] = akl;
        } else {
            Rq_cache[n+672] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[10];
        for (int ij = 0; ij < 10; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 14; ++batch_kl) {
            int task_kl0 = blockIdx.y * 224 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*14] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+224];
            double zkl = Rq_cache[sq_kl+448];
            double akl = Rq_cache[sq_kl+672];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 3, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 3; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+224] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+64];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+448] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+128];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+672] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_1_0 * dm_kl0;
            vj_ij[4] += R_0_0_1_1 * dm_kl0;
            vj_ij[5] += R_0_0_2_0 * dm_kl0;
            vj_ij[6] += R_0_1_0_0 * dm_kl0;
            vj_ij[7] += R_0_1_0_1 * dm_kl0;
            vj_ij[8] += R_0_1_1_0 * dm_kl0;
            vj_ij[9] += R_0_2_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+1];
            vj_ij[0] -= R_0_0_0_1 * dm_kl0;
            vj_ij[1] -= R_0_0_0_2 * dm_kl0;
            vj_ij[2] -= R_0_0_0_3 * dm_kl0;
            vj_ij[3] -= R_0_0_1_1 * dm_kl0;
            vj_ij[4] -= R_0_0_1_2 * dm_kl0;
            vj_ij[5] -= R_0_0_2_1 * dm_kl0;
            vj_ij[6] -= R_0_1_0_1 * dm_kl0;
            vj_ij[7] -= R_0_1_0_2 * dm_kl0;
            vj_ij[8] -= R_0_1_1_1 * dm_kl0;
            vj_ij[9] -= R_0_2_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+2];
            vj_ij[0] -= R_0_0_1_0 * dm_kl0;
            vj_ij[1] -= R_0_0_1_1 * dm_kl0;
            vj_ij[2] -= R_0_0_1_2 * dm_kl0;
            vj_ij[3] -= R_0_0_2_0 * dm_kl0;
            vj_ij[4] -= R_0_0_2_1 * dm_kl0;
            vj_ij[5] -= R_0_0_3_0 * dm_kl0;
            vj_ij[6] -= R_0_1_1_0 * dm_kl0;
            vj_ij[7] -= R_0_1_1_1 * dm_kl0;
            vj_ij[8] -= R_0_1_2_0 * dm_kl0;
            vj_ij[9] -= R_0_2_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+3];
            vj_ij[0] -= R_0_1_0_0 * dm_kl0;
            vj_ij[1] -= R_0_1_0_1 * dm_kl0;
            vj_ij[2] -= R_0_1_0_2 * dm_kl0;
            vj_ij[3] -= R_0_1_1_0 * dm_kl0;
            vj_ij[4] -= R_0_1_1_1 * dm_kl0;
            vj_ij[5] -= R_0_1_2_0 * dm_kl0;
            vj_ij[6] -= R_0_2_0_0 * dm_kl0;
            vj_ij[7] -= R_0_2_0_1 * dm_kl0;
            vj_ij[8] -= R_0_2_1_0 * dm_kl0;
            vj_ij[9] -= R_0_3_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 10; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 56; n += 16) {
        int batch_kl = n % 14;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 224 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 14;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*224]);
        }
    }
}

// TILEX=20, TILEY=20
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_2_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 320;
    int task_kl0 = blockIdx.y * 320;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+320 <= task_kl0) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 3200;
    double *Rp_cache = Rq_cache + 1280;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 4544; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 320; n += 256) {
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
        } else {
            Rq_cache[n+960] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 20; ++batch_ij) {
        int task_ij0 = blockIdx.x * 320 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[10];
        for (int ij = 0; ij < 10; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 20; ++batch_kl) {
            int task_kl0 = blockIdx.y * 320 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*20] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*20] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
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
            eval_gamma_inc_fn(gamma_inc, theta_rr, 4, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 4; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+320] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+16];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+48];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[tx+64];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+96];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[tx+112];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+128];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+640] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+64];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+960] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+64];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+112];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+128];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1280] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[tx+64];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+112];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[tx+128];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1600] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+128];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1920] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+128];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+2240] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+128];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+2560] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[tx+128];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+2880] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_1_0 * dm_kl0;
            vj_ij[4] += R_0_0_1_1 * dm_kl0;
            vj_ij[5] += R_0_0_2_0 * dm_kl0;
            vj_ij[6] += R_0_1_0_0 * dm_kl0;
            vj_ij[7] += R_0_1_0_1 * dm_kl0;
            vj_ij[8] += R_0_1_1_0 * dm_kl0;
            vj_ij[9] += R_0_2_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+1];
            vj_ij[0] -= R_0_0_0_1 * dm_kl0;
            vj_ij[1] -= R_0_0_0_2 * dm_kl0;
            vj_ij[2] -= R_0_0_0_3 * dm_kl0;
            vj_ij[3] -= R_0_0_1_1 * dm_kl0;
            vj_ij[4] -= R_0_0_1_2 * dm_kl0;
            vj_ij[5] -= R_0_0_2_1 * dm_kl0;
            vj_ij[6] -= R_0_1_0_1 * dm_kl0;
            vj_ij[7] -= R_0_1_0_2 * dm_kl0;
            vj_ij[8] -= R_0_1_1_1 * dm_kl0;
            vj_ij[9] -= R_0_2_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+2];
            vj_ij[0] += R_0_0_0_2 * dm_kl0;
            vj_ij[1] += R_0_0_0_3 * dm_kl0;
            vj_ij[2] += R_0_0_0_4 * dm_kl0;
            vj_ij[3] += R_0_0_1_2 * dm_kl0;
            vj_ij[4] += R_0_0_1_3 * dm_kl0;
            vj_ij[5] += R_0_0_2_2 * dm_kl0;
            vj_ij[6] += R_0_1_0_2 * dm_kl0;
            vj_ij[7] += R_0_1_0_3 * dm_kl0;
            vj_ij[8] += R_0_1_1_2 * dm_kl0;
            vj_ij[9] += R_0_2_0_2 * dm_kl0;
            dm_kl0 = dm[kl_loc0+3];
            vj_ij[0] -= R_0_0_1_0 * dm_kl0;
            vj_ij[1] -= R_0_0_1_1 * dm_kl0;
            vj_ij[2] -= R_0_0_1_2 * dm_kl0;
            vj_ij[3] -= R_0_0_2_0 * dm_kl0;
            vj_ij[4] -= R_0_0_2_1 * dm_kl0;
            vj_ij[5] -= R_0_0_3_0 * dm_kl0;
            vj_ij[6] -= R_0_1_1_0 * dm_kl0;
            vj_ij[7] -= R_0_1_1_1 * dm_kl0;
            vj_ij[8] -= R_0_1_2_0 * dm_kl0;
            vj_ij[9] -= R_0_2_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+4];
            vj_ij[0] += R_0_0_1_1 * dm_kl0;
            vj_ij[1] += R_0_0_1_2 * dm_kl0;
            vj_ij[2] += R_0_0_1_3 * dm_kl0;
            vj_ij[3] += R_0_0_2_1 * dm_kl0;
            vj_ij[4] += R_0_0_2_2 * dm_kl0;
            vj_ij[5] += R_0_0_3_1 * dm_kl0;
            vj_ij[6] += R_0_1_1_1 * dm_kl0;
            vj_ij[7] += R_0_1_1_2 * dm_kl0;
            vj_ij[8] += R_0_1_2_1 * dm_kl0;
            vj_ij[9] += R_0_2_1_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+5];
            vj_ij[0] += R_0_0_2_0 * dm_kl0;
            vj_ij[1] += R_0_0_2_1 * dm_kl0;
            vj_ij[2] += R_0_0_2_2 * dm_kl0;
            vj_ij[3] += R_0_0_3_0 * dm_kl0;
            vj_ij[4] += R_0_0_3_1 * dm_kl0;
            vj_ij[5] += R_0_0_4_0 * dm_kl0;
            vj_ij[6] += R_0_1_2_0 * dm_kl0;
            vj_ij[7] += R_0_1_2_1 * dm_kl0;
            vj_ij[8] += R_0_1_3_0 * dm_kl0;
            vj_ij[9] += R_0_2_2_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+6];
            vj_ij[0] -= R_0_1_0_0 * dm_kl0;
            vj_ij[1] -= R_0_1_0_1 * dm_kl0;
            vj_ij[2] -= R_0_1_0_2 * dm_kl0;
            vj_ij[3] -= R_0_1_1_0 * dm_kl0;
            vj_ij[4] -= R_0_1_1_1 * dm_kl0;
            vj_ij[5] -= R_0_1_2_0 * dm_kl0;
            vj_ij[6] -= R_0_2_0_0 * dm_kl0;
            vj_ij[7] -= R_0_2_0_1 * dm_kl0;
            vj_ij[8] -= R_0_2_1_0 * dm_kl0;
            vj_ij[9] -= R_0_3_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+7];
            vj_ij[0] += R_0_1_0_1 * dm_kl0;
            vj_ij[1] += R_0_1_0_2 * dm_kl0;
            vj_ij[2] += R_0_1_0_3 * dm_kl0;
            vj_ij[3] += R_0_1_1_1 * dm_kl0;
            vj_ij[4] += R_0_1_1_2 * dm_kl0;
            vj_ij[5] += R_0_1_2_1 * dm_kl0;
            vj_ij[6] += R_0_2_0_1 * dm_kl0;
            vj_ij[7] += R_0_2_0_2 * dm_kl0;
            vj_ij[8] += R_0_2_1_1 * dm_kl0;
            vj_ij[9] += R_0_3_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+8];
            vj_ij[0] += R_0_1_1_0 * dm_kl0;
            vj_ij[1] += R_0_1_1_1 * dm_kl0;
            vj_ij[2] += R_0_1_1_2 * dm_kl0;
            vj_ij[3] += R_0_1_2_0 * dm_kl0;
            vj_ij[4] += R_0_1_2_1 * dm_kl0;
            vj_ij[5] += R_0_1_3_0 * dm_kl0;
            vj_ij[6] += R_0_2_1_0 * dm_kl0;
            vj_ij[7] += R_0_2_1_1 * dm_kl0;
            vj_ij[8] += R_0_2_2_0 * dm_kl0;
            vj_ij[9] += R_0_3_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+9];
            vj_ij[0] += R_0_2_0_0 * dm_kl0;
            vj_ij[1] += R_0_2_0_1 * dm_kl0;
            vj_ij[2] += R_0_2_0_2 * dm_kl0;
            vj_ij[3] += R_0_2_1_0 * dm_kl0;
            vj_ij[4] += R_0_2_1_1 * dm_kl0;
            vj_ij[5] += R_0_2_2_0 * dm_kl0;
            vj_ij[6] += R_0_3_0_0 * dm_kl0;
            vj_ij[7] += R_0_3_0_1 * dm_kl0;
            vj_ij[8] += R_0_3_1_0 * dm_kl0;
            vj_ij[9] += R_0_4_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 10; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 200; n += 16) {
        int batch_kl = n % 20;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 320 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 20;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*320]);
        }
    }
}

// TILEX=48, TILEY=20
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_3_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 320;
    double *Rp_cache = Rq_cache + 1280;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 1664; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 320; n += 256) {
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
        } else {
            Rq_cache[n+960] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[20];
        for (int ij = 0; ij < 20; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 20; ++batch_kl) {
            int task_kl0 = blockIdx.y * 320 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*20] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
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
            eval_gamma_inc_fn(gamma_inc, theta_rr, 3, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 3; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+96];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+128];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+144];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+240];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_0_3 * dm_kl0;
            vj_ij[4] += R_0_0_1_0 * dm_kl0;
            vj_ij[5] += R_0_0_1_1 * dm_kl0;
            vj_ij[6] += R_0_0_1_2 * dm_kl0;
            vj_ij[7] += R_0_0_2_0 * dm_kl0;
            vj_ij[8] += R_0_0_2_1 * dm_kl0;
            vj_ij[9] += R_0_0_3_0 * dm_kl0;
            vj_ij[10] += R_0_1_0_0 * dm_kl0;
            vj_ij[11] += R_0_1_0_1 * dm_kl0;
            vj_ij[12] += R_0_1_0_2 * dm_kl0;
            vj_ij[13] += R_0_1_1_0 * dm_kl0;
            vj_ij[14] += R_0_1_1_1 * dm_kl0;
            vj_ij[15] += R_0_1_2_0 * dm_kl0;
            vj_ij[16] += R_0_2_0_0 * dm_kl0;
            vj_ij[17] += R_0_2_0_1 * dm_kl0;
            vj_ij[18] += R_0_2_1_0 * dm_kl0;
            vj_ij[19] += R_0_3_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 20; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 20; n += 16) {
        int batch_kl = n % 20;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 320 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 20;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*320]);
        }
    }
}

// TILEX=48, TILEY=35
__global__ static
void md_j_3_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 560;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2240;
    double *Rp_cache = Rq_cache + 2240;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 4544; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 560; n += 256) {
        int task_kl = blockIdx.y * 560 + n;
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
            Rq_cache[n+560] = ykl;
            Rq_cache[n+1120] = zkl;
            Rq_cache[n+1680] = akl;
        } else {
            Rq_cache[n+1680] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[20];
        for (int ij = 0; ij < 20; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 35; ++batch_kl) {
            int task_kl0 = blockIdx.y * 560 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*35] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+560];
            double zkl = Rq_cache[sq_kl+1120];
            double akl = Rq_cache[sq_kl+1680];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 4, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 4; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+96];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+128];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+144];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+240];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[tx+32];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 -= R_0_0_0_4 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+80];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+112];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[tx+128];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+176];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+208];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+224];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+256];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[tx+272];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+560] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[tx+128];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 -= R_0_0_4_0 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+224];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+272];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[tx+288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1120] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[tx+288];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 -= R_0_4_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1680] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_0_3 * dm_kl0;
            vj_ij[4] += R_0_0_1_0 * dm_kl0;
            vj_ij[5] += R_0_0_1_1 * dm_kl0;
            vj_ij[6] += R_0_0_1_2 * dm_kl0;
            vj_ij[7] += R_0_0_2_0 * dm_kl0;
            vj_ij[8] += R_0_0_2_1 * dm_kl0;
            vj_ij[9] += R_0_0_3_0 * dm_kl0;
            vj_ij[10] += R_0_1_0_0 * dm_kl0;
            vj_ij[11] += R_0_1_0_1 * dm_kl0;
            vj_ij[12] += R_0_1_0_2 * dm_kl0;
            vj_ij[13] += R_0_1_1_0 * dm_kl0;
            vj_ij[14] += R_0_1_1_1 * dm_kl0;
            vj_ij[15] += R_0_1_2_0 * dm_kl0;
            vj_ij[16] += R_0_2_0_0 * dm_kl0;
            vj_ij[17] += R_0_2_0_1 * dm_kl0;
            vj_ij[18] += R_0_2_1_0 * dm_kl0;
            vj_ij[19] += R_0_3_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+1];
            vj_ij[0] -= R_0_0_0_1 * dm_kl0;
            vj_ij[1] -= R_0_0_0_2 * dm_kl0;
            vj_ij[2] -= R_0_0_0_3 * dm_kl0;
            vj_ij[3] -= R_0_0_0_4 * dm_kl0;
            vj_ij[4] -= R_0_0_1_1 * dm_kl0;
            vj_ij[5] -= R_0_0_1_2 * dm_kl0;
            vj_ij[6] -= R_0_0_1_3 * dm_kl0;
            vj_ij[7] -= R_0_0_2_1 * dm_kl0;
            vj_ij[8] -= R_0_0_2_2 * dm_kl0;
            vj_ij[9] -= R_0_0_3_1 * dm_kl0;
            vj_ij[10] -= R_0_1_0_1 * dm_kl0;
            vj_ij[11] -= R_0_1_0_2 * dm_kl0;
            vj_ij[12] -= R_0_1_0_3 * dm_kl0;
            vj_ij[13] -= R_0_1_1_1 * dm_kl0;
            vj_ij[14] -= R_0_1_1_2 * dm_kl0;
            vj_ij[15] -= R_0_1_2_1 * dm_kl0;
            vj_ij[16] -= R_0_2_0_1 * dm_kl0;
            vj_ij[17] -= R_0_2_0_2 * dm_kl0;
            vj_ij[18] -= R_0_2_1_1 * dm_kl0;
            vj_ij[19] -= R_0_3_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+2];
            vj_ij[0] -= R_0_0_1_0 * dm_kl0;
            vj_ij[1] -= R_0_0_1_1 * dm_kl0;
            vj_ij[2] -= R_0_0_1_2 * dm_kl0;
            vj_ij[3] -= R_0_0_1_3 * dm_kl0;
            vj_ij[4] -= R_0_0_2_0 * dm_kl0;
            vj_ij[5] -= R_0_0_2_1 * dm_kl0;
            vj_ij[6] -= R_0_0_2_2 * dm_kl0;
            vj_ij[7] -= R_0_0_3_0 * dm_kl0;
            vj_ij[8] -= R_0_0_3_1 * dm_kl0;
            vj_ij[9] -= R_0_0_4_0 * dm_kl0;
            vj_ij[10] -= R_0_1_1_0 * dm_kl0;
            vj_ij[11] -= R_0_1_1_1 * dm_kl0;
            vj_ij[12] -= R_0_1_1_2 * dm_kl0;
            vj_ij[13] -= R_0_1_2_0 * dm_kl0;
            vj_ij[14] -= R_0_1_2_1 * dm_kl0;
            vj_ij[15] -= R_0_1_3_0 * dm_kl0;
            vj_ij[16] -= R_0_2_1_0 * dm_kl0;
            vj_ij[17] -= R_0_2_1_1 * dm_kl0;
            vj_ij[18] -= R_0_2_2_0 * dm_kl0;
            vj_ij[19] -= R_0_3_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+3];
            vj_ij[0] -= R_0_1_0_0 * dm_kl0;
            vj_ij[1] -= R_0_1_0_1 * dm_kl0;
            vj_ij[2] -= R_0_1_0_2 * dm_kl0;
            vj_ij[3] -= R_0_1_0_3 * dm_kl0;
            vj_ij[4] -= R_0_1_1_0 * dm_kl0;
            vj_ij[5] -= R_0_1_1_1 * dm_kl0;
            vj_ij[6] -= R_0_1_1_2 * dm_kl0;
            vj_ij[7] -= R_0_1_2_0 * dm_kl0;
            vj_ij[8] -= R_0_1_2_1 * dm_kl0;
            vj_ij[9] -= R_0_1_3_0 * dm_kl0;
            vj_ij[10] -= R_0_2_0_0 * dm_kl0;
            vj_ij[11] -= R_0_2_0_1 * dm_kl0;
            vj_ij[12] -= R_0_2_0_2 * dm_kl0;
            vj_ij[13] -= R_0_2_1_0 * dm_kl0;
            vj_ij[14] -= R_0_2_1_1 * dm_kl0;
            vj_ij[15] -= R_0_2_2_0 * dm_kl0;
            vj_ij[16] -= R_0_3_0_0 * dm_kl0;
            vj_ij[17] -= R_0_3_0_1 * dm_kl0;
            vj_ij[18] -= R_0_3_1_0 * dm_kl0;
            vj_ij[19] -= R_0_4_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 20; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 140; n += 16) {
        int batch_kl = n % 35;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 560 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 35;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*560]);
        }
    }
}

// TILEX=48, TILEY=18
__global__ static
void md_j_3_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 288;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2880;
    double *Rp_cache = Rq_cache + 1152;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 4096; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 288; n += 256) {
        int task_kl = blockIdx.y * 288 + n;
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
            Rq_cache[n+288] = ykl;
            Rq_cache[n+576] = zkl;
            Rq_cache[n+864] = akl;
        } else {
            Rq_cache[n+864] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[20];
        for (int ij = 0; ij < 20; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 18; ++batch_kl) {
            int task_kl0 = blockIdx.y * 288 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*18] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+288];
            double zkl = Rq_cache[sq_kl+576];
            double akl = Rq_cache[sq_kl+864];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 5, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 5; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+96];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+128];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+144];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+240];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[tx+32];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 -= R_0_0_0_4 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+80];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+112];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[tx+128];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+176];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+208];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+224];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+256];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[tx+272];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+288] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[tx+32];
            double R_4_0_0_1 = zpq * gamma_inc[sq_id+5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl0 += R_0_0_0_5 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[tx+80];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl0 += R_0_0_1_4 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+112];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[tx+128];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[tx+144];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+160];
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[tx+176];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl0 += R_0_1_0_4 * dm_ij_cache[tx+192];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+208];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[tx+224];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[tx+240];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+256];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[tx+272];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[tx+288];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+576] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[tx+128];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 -= R_0_0_4_0 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+224];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+272];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[tx+288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+864] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_0_1_4 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[tx+128];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl0 += R_0_0_4_1 * dm_ij_cache[tx+144];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+160];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+176];
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[tx+192];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+208];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[tx+224];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[tx+240];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+256];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[tx+272];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[tx+288];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1152] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_0_4_1 * dm_ij_cache[tx+128];
            double R_4_0_1_0 = ypq * gamma_inc[sq_id+5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl0 += R_0_0_5_0 * dm_ij_cache[tx+144];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+160];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+176];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[tx+192];
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[tx+208];
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[tx+224];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl0 += R_0_1_4_0 * dm_ij_cache[tx+240];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+256];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[tx+272];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[tx+288];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1440] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[tx+288];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 -= R_0_4_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1728] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_1_0_4 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[tx+128];
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[tx+144];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+160];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+176];
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[tx+192];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+208];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[tx+224];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[tx+240];
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[tx+256];
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[tx+272];
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[tx+288];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl0 += R_0_4_0_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+2016] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[tx+128];
            vj_kl0 += R_0_1_4_0 * dm_ij_cache[tx+144];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+160];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+176];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[tx+192];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+208];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[tx+224];
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[tx+240];
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[tx+256];
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[tx+272];
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[tx+288];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl0 += R_0_4_1_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+2304] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+0];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+16];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+32];
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[tx+48];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+64];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+80];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[tx+96];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+112];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[tx+128];
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[tx+144];
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+160];
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[tx+176];
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[tx+192];
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[tx+208];
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[tx+224];
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[tx+240];
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[tx+256];
            vj_kl0 += R_0_4_0_1 * dm_ij_cache[tx+272];
            vj_kl0 += R_0_4_1_0 * dm_ij_cache[tx+288];
            double R_4_1_0_0 = xpq * gamma_inc[sq_id+5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl0 += R_0_5_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+2592] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_0_3 * dm_kl0;
            vj_ij[4] += R_0_0_1_0 * dm_kl0;
            vj_ij[5] += R_0_0_1_1 * dm_kl0;
            vj_ij[6] += R_0_0_1_2 * dm_kl0;
            vj_ij[7] += R_0_0_2_0 * dm_kl0;
            vj_ij[8] += R_0_0_2_1 * dm_kl0;
            vj_ij[9] += R_0_0_3_0 * dm_kl0;
            vj_ij[10] += R_0_1_0_0 * dm_kl0;
            vj_ij[11] += R_0_1_0_1 * dm_kl0;
            vj_ij[12] += R_0_1_0_2 * dm_kl0;
            vj_ij[13] += R_0_1_1_0 * dm_kl0;
            vj_ij[14] += R_0_1_1_1 * dm_kl0;
            vj_ij[15] += R_0_1_2_0 * dm_kl0;
            vj_ij[16] += R_0_2_0_0 * dm_kl0;
            vj_ij[17] += R_0_2_0_1 * dm_kl0;
            vj_ij[18] += R_0_2_1_0 * dm_kl0;
            vj_ij[19] += R_0_3_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+1];
            vj_ij[0] -= R_0_0_0_1 * dm_kl0;
            vj_ij[1] -= R_0_0_0_2 * dm_kl0;
            vj_ij[2] -= R_0_0_0_3 * dm_kl0;
            vj_ij[3] -= R_0_0_0_4 * dm_kl0;
            vj_ij[4] -= R_0_0_1_1 * dm_kl0;
            vj_ij[5] -= R_0_0_1_2 * dm_kl0;
            vj_ij[6] -= R_0_0_1_3 * dm_kl0;
            vj_ij[7] -= R_0_0_2_1 * dm_kl0;
            vj_ij[8] -= R_0_0_2_2 * dm_kl0;
            vj_ij[9] -= R_0_0_3_1 * dm_kl0;
            vj_ij[10] -= R_0_1_0_1 * dm_kl0;
            vj_ij[11] -= R_0_1_0_2 * dm_kl0;
            vj_ij[12] -= R_0_1_0_3 * dm_kl0;
            vj_ij[13] -= R_0_1_1_1 * dm_kl0;
            vj_ij[14] -= R_0_1_1_2 * dm_kl0;
            vj_ij[15] -= R_0_1_2_1 * dm_kl0;
            vj_ij[16] -= R_0_2_0_1 * dm_kl0;
            vj_ij[17] -= R_0_2_0_2 * dm_kl0;
            vj_ij[18] -= R_0_2_1_1 * dm_kl0;
            vj_ij[19] -= R_0_3_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+2];
            vj_ij[0] += R_0_0_0_2 * dm_kl0;
            vj_ij[1] += R_0_0_0_3 * dm_kl0;
            vj_ij[2] += R_0_0_0_4 * dm_kl0;
            vj_ij[3] += R_0_0_0_5 * dm_kl0;
            vj_ij[4] += R_0_0_1_2 * dm_kl0;
            vj_ij[5] += R_0_0_1_3 * dm_kl0;
            vj_ij[6] += R_0_0_1_4 * dm_kl0;
            vj_ij[7] += R_0_0_2_2 * dm_kl0;
            vj_ij[8] += R_0_0_2_3 * dm_kl0;
            vj_ij[9] += R_0_0_3_2 * dm_kl0;
            vj_ij[10] += R_0_1_0_2 * dm_kl0;
            vj_ij[11] += R_0_1_0_3 * dm_kl0;
            vj_ij[12] += R_0_1_0_4 * dm_kl0;
            vj_ij[13] += R_0_1_1_2 * dm_kl0;
            vj_ij[14] += R_0_1_1_3 * dm_kl0;
            vj_ij[15] += R_0_1_2_2 * dm_kl0;
            vj_ij[16] += R_0_2_0_2 * dm_kl0;
            vj_ij[17] += R_0_2_0_3 * dm_kl0;
            vj_ij[18] += R_0_2_1_2 * dm_kl0;
            vj_ij[19] += R_0_3_0_2 * dm_kl0;
            dm_kl0 = dm[kl_loc0+3];
            vj_ij[0] -= R_0_0_1_0 * dm_kl0;
            vj_ij[1] -= R_0_0_1_1 * dm_kl0;
            vj_ij[2] -= R_0_0_1_2 * dm_kl0;
            vj_ij[3] -= R_0_0_1_3 * dm_kl0;
            vj_ij[4] -= R_0_0_2_0 * dm_kl0;
            vj_ij[5] -= R_0_0_2_1 * dm_kl0;
            vj_ij[6] -= R_0_0_2_2 * dm_kl0;
            vj_ij[7] -= R_0_0_3_0 * dm_kl0;
            vj_ij[8] -= R_0_0_3_1 * dm_kl0;
            vj_ij[9] -= R_0_0_4_0 * dm_kl0;
            vj_ij[10] -= R_0_1_1_0 * dm_kl0;
            vj_ij[11] -= R_0_1_1_1 * dm_kl0;
            vj_ij[12] -= R_0_1_1_2 * dm_kl0;
            vj_ij[13] -= R_0_1_2_0 * dm_kl0;
            vj_ij[14] -= R_0_1_2_1 * dm_kl0;
            vj_ij[15] -= R_0_1_3_0 * dm_kl0;
            vj_ij[16] -= R_0_2_1_0 * dm_kl0;
            vj_ij[17] -= R_0_2_1_1 * dm_kl0;
            vj_ij[18] -= R_0_2_2_0 * dm_kl0;
            vj_ij[19] -= R_0_3_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+4];
            vj_ij[0] += R_0_0_1_1 * dm_kl0;
            vj_ij[1] += R_0_0_1_2 * dm_kl0;
            vj_ij[2] += R_0_0_1_3 * dm_kl0;
            vj_ij[3] += R_0_0_1_4 * dm_kl0;
            vj_ij[4] += R_0_0_2_1 * dm_kl0;
            vj_ij[5] += R_0_0_2_2 * dm_kl0;
            vj_ij[6] += R_0_0_2_3 * dm_kl0;
            vj_ij[7] += R_0_0_3_1 * dm_kl0;
            vj_ij[8] += R_0_0_3_2 * dm_kl0;
            vj_ij[9] += R_0_0_4_1 * dm_kl0;
            vj_ij[10] += R_0_1_1_1 * dm_kl0;
            vj_ij[11] += R_0_1_1_2 * dm_kl0;
            vj_ij[12] += R_0_1_1_3 * dm_kl0;
            vj_ij[13] += R_0_1_2_1 * dm_kl0;
            vj_ij[14] += R_0_1_2_2 * dm_kl0;
            vj_ij[15] += R_0_1_3_1 * dm_kl0;
            vj_ij[16] += R_0_2_1_1 * dm_kl0;
            vj_ij[17] += R_0_2_1_2 * dm_kl0;
            vj_ij[18] += R_0_2_2_1 * dm_kl0;
            vj_ij[19] += R_0_3_1_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+5];
            vj_ij[0] += R_0_0_2_0 * dm_kl0;
            vj_ij[1] += R_0_0_2_1 * dm_kl0;
            vj_ij[2] += R_0_0_2_2 * dm_kl0;
            vj_ij[3] += R_0_0_2_3 * dm_kl0;
            vj_ij[4] += R_0_0_3_0 * dm_kl0;
            vj_ij[5] += R_0_0_3_1 * dm_kl0;
            vj_ij[6] += R_0_0_3_2 * dm_kl0;
            vj_ij[7] += R_0_0_4_0 * dm_kl0;
            vj_ij[8] += R_0_0_4_1 * dm_kl0;
            vj_ij[9] += R_0_0_5_0 * dm_kl0;
            vj_ij[10] += R_0_1_2_0 * dm_kl0;
            vj_ij[11] += R_0_1_2_1 * dm_kl0;
            vj_ij[12] += R_0_1_2_2 * dm_kl0;
            vj_ij[13] += R_0_1_3_0 * dm_kl0;
            vj_ij[14] += R_0_1_3_1 * dm_kl0;
            vj_ij[15] += R_0_1_4_0 * dm_kl0;
            vj_ij[16] += R_0_2_2_0 * dm_kl0;
            vj_ij[17] += R_0_2_2_1 * dm_kl0;
            vj_ij[18] += R_0_2_3_0 * dm_kl0;
            vj_ij[19] += R_0_3_2_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+6];
            vj_ij[0] -= R_0_1_0_0 * dm_kl0;
            vj_ij[1] -= R_0_1_0_1 * dm_kl0;
            vj_ij[2] -= R_0_1_0_2 * dm_kl0;
            vj_ij[3] -= R_0_1_0_3 * dm_kl0;
            vj_ij[4] -= R_0_1_1_0 * dm_kl0;
            vj_ij[5] -= R_0_1_1_1 * dm_kl0;
            vj_ij[6] -= R_0_1_1_2 * dm_kl0;
            vj_ij[7] -= R_0_1_2_0 * dm_kl0;
            vj_ij[8] -= R_0_1_2_1 * dm_kl0;
            vj_ij[9] -= R_0_1_3_0 * dm_kl0;
            vj_ij[10] -= R_0_2_0_0 * dm_kl0;
            vj_ij[11] -= R_0_2_0_1 * dm_kl0;
            vj_ij[12] -= R_0_2_0_2 * dm_kl0;
            vj_ij[13] -= R_0_2_1_0 * dm_kl0;
            vj_ij[14] -= R_0_2_1_1 * dm_kl0;
            vj_ij[15] -= R_0_2_2_0 * dm_kl0;
            vj_ij[16] -= R_0_3_0_0 * dm_kl0;
            vj_ij[17] -= R_0_3_0_1 * dm_kl0;
            vj_ij[18] -= R_0_3_1_0 * dm_kl0;
            vj_ij[19] -= R_0_4_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+7];
            vj_ij[0] += R_0_1_0_1 * dm_kl0;
            vj_ij[1] += R_0_1_0_2 * dm_kl0;
            vj_ij[2] += R_0_1_0_3 * dm_kl0;
            vj_ij[3] += R_0_1_0_4 * dm_kl0;
            vj_ij[4] += R_0_1_1_1 * dm_kl0;
            vj_ij[5] += R_0_1_1_2 * dm_kl0;
            vj_ij[6] += R_0_1_1_3 * dm_kl0;
            vj_ij[7] += R_0_1_2_1 * dm_kl0;
            vj_ij[8] += R_0_1_2_2 * dm_kl0;
            vj_ij[9] += R_0_1_3_1 * dm_kl0;
            vj_ij[10] += R_0_2_0_1 * dm_kl0;
            vj_ij[11] += R_0_2_0_2 * dm_kl0;
            vj_ij[12] += R_0_2_0_3 * dm_kl0;
            vj_ij[13] += R_0_2_1_1 * dm_kl0;
            vj_ij[14] += R_0_2_1_2 * dm_kl0;
            vj_ij[15] += R_0_2_2_1 * dm_kl0;
            vj_ij[16] += R_0_3_0_1 * dm_kl0;
            vj_ij[17] += R_0_3_0_2 * dm_kl0;
            vj_ij[18] += R_0_3_1_1 * dm_kl0;
            vj_ij[19] += R_0_4_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+8];
            vj_ij[0] += R_0_1_1_0 * dm_kl0;
            vj_ij[1] += R_0_1_1_1 * dm_kl0;
            vj_ij[2] += R_0_1_1_2 * dm_kl0;
            vj_ij[3] += R_0_1_1_3 * dm_kl0;
            vj_ij[4] += R_0_1_2_0 * dm_kl0;
            vj_ij[5] += R_0_1_2_1 * dm_kl0;
            vj_ij[6] += R_0_1_2_2 * dm_kl0;
            vj_ij[7] += R_0_1_3_0 * dm_kl0;
            vj_ij[8] += R_0_1_3_1 * dm_kl0;
            vj_ij[9] += R_0_1_4_0 * dm_kl0;
            vj_ij[10] += R_0_2_1_0 * dm_kl0;
            vj_ij[11] += R_0_2_1_1 * dm_kl0;
            vj_ij[12] += R_0_2_1_2 * dm_kl0;
            vj_ij[13] += R_0_2_2_0 * dm_kl0;
            vj_ij[14] += R_0_2_2_1 * dm_kl0;
            vj_ij[15] += R_0_2_3_0 * dm_kl0;
            vj_ij[16] += R_0_3_1_0 * dm_kl0;
            vj_ij[17] += R_0_3_1_1 * dm_kl0;
            vj_ij[18] += R_0_3_2_0 * dm_kl0;
            vj_ij[19] += R_0_4_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+9];
            vj_ij[0] += R_0_2_0_0 * dm_kl0;
            vj_ij[1] += R_0_2_0_1 * dm_kl0;
            vj_ij[2] += R_0_2_0_2 * dm_kl0;
            vj_ij[3] += R_0_2_0_3 * dm_kl0;
            vj_ij[4] += R_0_2_1_0 * dm_kl0;
            vj_ij[5] += R_0_2_1_1 * dm_kl0;
            vj_ij[6] += R_0_2_1_2 * dm_kl0;
            vj_ij[7] += R_0_2_2_0 * dm_kl0;
            vj_ij[8] += R_0_2_2_1 * dm_kl0;
            vj_ij[9] += R_0_2_3_0 * dm_kl0;
            vj_ij[10] += R_0_3_0_0 * dm_kl0;
            vj_ij[11] += R_0_3_0_1 * dm_kl0;
            vj_ij[12] += R_0_3_0_2 * dm_kl0;
            vj_ij[13] += R_0_3_1_0 * dm_kl0;
            vj_ij[14] += R_0_3_1_1 * dm_kl0;
            vj_ij[15] += R_0_3_2_0 * dm_kl0;
            vj_ij[16] += R_0_4_0_0 * dm_kl0;
            vj_ij[17] += R_0_4_0_1 * dm_kl0;
            vj_ij[18] += R_0_4_1_0 * dm_kl0;
            vj_ij[19] += R_0_5_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 20; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 180; n += 16) {
        int batch_kl = n % 18;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 288 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 18;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*288]);
        }
    }
}

// TILEX=48, TILEY=48
__global__ static
void md_j_4_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 768;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 768;
    double *Rp_cache = Rq_cache + 3072;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 560;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3904; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 768; n += 256) {
        int task_kl = blockIdx.y * 768 + n;
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
            Rq_cache[n+768] = ykl;
            Rq_cache[n+1536] = zkl;
            Rq_cache[n+2304] = akl;
        } else {
            Rq_cache[n+2304] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[35];
        for (int ij = 0; ij < 35; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 48; ++batch_kl) {
            int task_kl0 = blockIdx.y * 768 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*48] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+768];
            double zkl = Rq_cache[sq_kl+1536];
            double akl = Rq_cache[sq_kl+2304];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 4, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 4; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[tx+64];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+96];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+112];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[tx+128];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+144];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+160];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+176];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+192];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[tx+208];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[tx+224];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+240];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+272];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[tx+288];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+304];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+320];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+336];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+352];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+368];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[tx+384];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+400];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+416];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+432];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+448];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+464];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+480];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+496];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[tx+512];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[tx+528];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[tx+544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_0_3 * dm_kl0;
            vj_ij[4] += R_0_0_0_4 * dm_kl0;
            vj_ij[5] += R_0_0_1_0 * dm_kl0;
            vj_ij[6] += R_0_0_1_1 * dm_kl0;
            vj_ij[7] += R_0_0_1_2 * dm_kl0;
            vj_ij[8] += R_0_0_1_3 * dm_kl0;
            vj_ij[9] += R_0_0_2_0 * dm_kl0;
            vj_ij[10] += R_0_0_2_1 * dm_kl0;
            vj_ij[11] += R_0_0_2_2 * dm_kl0;
            vj_ij[12] += R_0_0_3_0 * dm_kl0;
            vj_ij[13] += R_0_0_3_1 * dm_kl0;
            vj_ij[14] += R_0_0_4_0 * dm_kl0;
            vj_ij[15] += R_0_1_0_0 * dm_kl0;
            vj_ij[16] += R_0_1_0_1 * dm_kl0;
            vj_ij[17] += R_0_1_0_2 * dm_kl0;
            vj_ij[18] += R_0_1_0_3 * dm_kl0;
            vj_ij[19] += R_0_1_1_0 * dm_kl0;
            vj_ij[20] += R_0_1_1_1 * dm_kl0;
            vj_ij[21] += R_0_1_1_2 * dm_kl0;
            vj_ij[22] += R_0_1_2_0 * dm_kl0;
            vj_ij[23] += R_0_1_2_1 * dm_kl0;
            vj_ij[24] += R_0_1_3_0 * dm_kl0;
            vj_ij[25] += R_0_2_0_0 * dm_kl0;
            vj_ij[26] += R_0_2_0_1 * dm_kl0;
            vj_ij[27] += R_0_2_0_2 * dm_kl0;
            vj_ij[28] += R_0_2_1_0 * dm_kl0;
            vj_ij[29] += R_0_2_1_1 * dm_kl0;
            vj_ij[30] += R_0_2_2_0 * dm_kl0;
            vj_ij[31] += R_0_3_0_0 * dm_kl0;
            vj_ij[32] += R_0_3_0_1 * dm_kl0;
            vj_ij[33] += R_0_3_1_0 * dm_kl0;
            vj_ij[34] += R_0_4_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 35; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 48; n += 16) {
        int batch_kl = n % 48;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 768 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 48;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*768]);
        }
    }
}

// TILEX=48, TILEY=31
__global__ static
void md_j_4_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 496;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1984;
    double *Rp_cache = Rq_cache + 1984;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 560;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 4032; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 496; n += 256) {
        int task_kl = blockIdx.y * 496 + n;
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
            Rq_cache[n+496] = ykl;
            Rq_cache[n+992] = zkl;
            Rq_cache[n+1488] = akl;
        } else {
            Rq_cache[n+1488] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[35];
        for (int ij = 0; ij < 35; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 31; ++batch_kl) {
            int task_kl0 = blockIdx.y * 496 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*31] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+496];
            double zkl = Rq_cache[sq_kl+992];
            double akl = Rq_cache[sq_kl+1488];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 5, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 5; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[tx+64];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+96];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+112];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[tx+128];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+144];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+160];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+176];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+192];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[tx+208];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[tx+224];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+240];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+272];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[tx+288];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+304];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+320];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+336];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+352];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+368];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[tx+384];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+400];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+416];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+432];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+448];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+464];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+480];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+496];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[tx+512];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[tx+528];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[tx+544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_0_4 * dm_ij_cache[tx+48];
            double R_4_0_0_1 = zpq * gamma_inc[sq_id+5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl0 -= R_0_0_0_5 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[tx+112];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl0 -= R_0_0_1_4 * dm_ij_cache[tx+128];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[tx+160];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl0 -= R_0_0_2_3 * dm_ij_cache[tx+176];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[tx+192];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl0 -= R_0_0_3_2 * dm_ij_cache[tx+208];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl0 -= R_0_0_4_1 * dm_ij_cache[tx+224];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+256];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[tx+272];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl0 -= R_0_1_0_4 * dm_ij_cache[tx+288];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+304];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+320];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl0 -= R_0_1_1_3 * dm_ij_cache[tx+336];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+352];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl0 -= R_0_1_2_2 * dm_ij_cache[tx+368];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl0 -= R_0_1_3_1 * dm_ij_cache[tx+384];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+400];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[tx+416];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl0 -= R_0_2_0_3 * dm_ij_cache[tx+432];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+448];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl0 -= R_0_2_1_2 * dm_ij_cache[tx+464];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl0 -= R_0_2_2_1 * dm_ij_cache[tx+480];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[tx+496];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl0 -= R_0_3_0_2 * dm_ij_cache[tx+512];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl0 -= R_0_3_1_1 * dm_ij_cache[tx+528];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl0 -= R_0_4_0_1 * dm_ij_cache[tx+544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+496] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_0_1_4 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_0_2_3 * dm_ij_cache[tx+128];
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_0_3_2 * dm_ij_cache[tx+176];
            vj_kl0 -= R_0_0_4_0 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_0_4_1 * dm_ij_cache[tx+208];
            double R_4_0_1_0 = ypq * gamma_inc[sq_id+5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl0 -= R_0_0_5_0 * dm_ij_cache[tx+224];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+256];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+272];
            vj_kl0 -= R_0_1_1_3 * dm_ij_cache[tx+288];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+304];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+320];
            vj_kl0 -= R_0_1_2_2 * dm_ij_cache[tx+336];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[tx+352];
            vj_kl0 -= R_0_1_3_1 * dm_ij_cache[tx+368];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl0 -= R_0_1_4_0 * dm_ij_cache[tx+384];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+400];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+416];
            vj_kl0 -= R_0_2_1_2 * dm_ij_cache[tx+432];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[tx+448];
            vj_kl0 -= R_0_2_2_1 * dm_ij_cache[tx+464];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl0 -= R_0_2_3_0 * dm_ij_cache[tx+480];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[tx+496];
            vj_kl0 -= R_0_3_1_1 * dm_ij_cache[tx+512];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl0 -= R_0_3_2_0 * dm_ij_cache[tx+528];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl0 -= R_0_4_1_0 * dm_ij_cache[tx+544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+992] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[tx+48];
            vj_kl0 -= R_0_1_0_4 * dm_ij_cache[tx+64];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[tx+80];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[tx+96];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[tx+112];
            vj_kl0 -= R_0_1_1_3 * dm_ij_cache[tx+128];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[tx+144];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[tx+160];
            vj_kl0 -= R_0_1_2_2 * dm_ij_cache[tx+176];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[tx+192];
            vj_kl0 -= R_0_1_3_1 * dm_ij_cache[tx+208];
            vj_kl0 -= R_0_1_4_0 * dm_ij_cache[tx+224];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[tx+240];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[tx+256];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[tx+272];
            vj_kl0 -= R_0_2_0_3 * dm_ij_cache[tx+288];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[tx+304];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[tx+320];
            vj_kl0 -= R_0_2_1_2 * dm_ij_cache[tx+336];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[tx+352];
            vj_kl0 -= R_0_2_2_1 * dm_ij_cache[tx+368];
            vj_kl0 -= R_0_2_3_0 * dm_ij_cache[tx+384];
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[tx+400];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[tx+416];
            vj_kl0 -= R_0_3_0_2 * dm_ij_cache[tx+432];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[tx+448];
            vj_kl0 -= R_0_3_1_1 * dm_ij_cache[tx+464];
            vj_kl0 -= R_0_3_2_0 * dm_ij_cache[tx+480];
            vj_kl0 -= R_0_4_0_0 * dm_ij_cache[tx+496];
            vj_kl0 -= R_0_4_0_1 * dm_ij_cache[tx+512];
            vj_kl0 -= R_0_4_1_0 * dm_ij_cache[tx+528];
            double R_4_1_0_0 = xpq * gamma_inc[sq_id+5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl0 -= R_0_5_0_0 * dm_ij_cache[tx+544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1488] += vj_kl0; }
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_0_3 * dm_kl0;
            vj_ij[4] += R_0_0_0_4 * dm_kl0;
            vj_ij[5] += R_0_0_1_0 * dm_kl0;
            vj_ij[6] += R_0_0_1_1 * dm_kl0;
            vj_ij[7] += R_0_0_1_2 * dm_kl0;
            vj_ij[8] += R_0_0_1_3 * dm_kl0;
            vj_ij[9] += R_0_0_2_0 * dm_kl0;
            vj_ij[10] += R_0_0_2_1 * dm_kl0;
            vj_ij[11] += R_0_0_2_2 * dm_kl0;
            vj_ij[12] += R_0_0_3_0 * dm_kl0;
            vj_ij[13] += R_0_0_3_1 * dm_kl0;
            vj_ij[14] += R_0_0_4_0 * dm_kl0;
            vj_ij[15] += R_0_1_0_0 * dm_kl0;
            vj_ij[16] += R_0_1_0_1 * dm_kl0;
            vj_ij[17] += R_0_1_0_2 * dm_kl0;
            vj_ij[18] += R_0_1_0_3 * dm_kl0;
            vj_ij[19] += R_0_1_1_0 * dm_kl0;
            vj_ij[20] += R_0_1_1_1 * dm_kl0;
            vj_ij[21] += R_0_1_1_2 * dm_kl0;
            vj_ij[22] += R_0_1_2_0 * dm_kl0;
            vj_ij[23] += R_0_1_2_1 * dm_kl0;
            vj_ij[24] += R_0_1_3_0 * dm_kl0;
            vj_ij[25] += R_0_2_0_0 * dm_kl0;
            vj_ij[26] += R_0_2_0_1 * dm_kl0;
            vj_ij[27] += R_0_2_0_2 * dm_kl0;
            vj_ij[28] += R_0_2_1_0 * dm_kl0;
            vj_ij[29] += R_0_2_1_1 * dm_kl0;
            vj_ij[30] += R_0_2_2_0 * dm_kl0;
            vj_ij[31] += R_0_3_0_0 * dm_kl0;
            vj_ij[32] += R_0_3_0_1 * dm_kl0;
            vj_ij[33] += R_0_3_1_0 * dm_kl0;
            vj_ij[34] += R_0_4_0_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+1];
            vj_ij[0] -= R_0_0_0_1 * dm_kl0;
            vj_ij[1] -= R_0_0_0_2 * dm_kl0;
            vj_ij[2] -= R_0_0_0_3 * dm_kl0;
            vj_ij[3] -= R_0_0_0_4 * dm_kl0;
            vj_ij[4] -= R_0_0_0_5 * dm_kl0;
            vj_ij[5] -= R_0_0_1_1 * dm_kl0;
            vj_ij[6] -= R_0_0_1_2 * dm_kl0;
            vj_ij[7] -= R_0_0_1_3 * dm_kl0;
            vj_ij[8] -= R_0_0_1_4 * dm_kl0;
            vj_ij[9] -= R_0_0_2_1 * dm_kl0;
            vj_ij[10] -= R_0_0_2_2 * dm_kl0;
            vj_ij[11] -= R_0_0_2_3 * dm_kl0;
            vj_ij[12] -= R_0_0_3_1 * dm_kl0;
            vj_ij[13] -= R_0_0_3_2 * dm_kl0;
            vj_ij[14] -= R_0_0_4_1 * dm_kl0;
            vj_ij[15] -= R_0_1_0_1 * dm_kl0;
            vj_ij[16] -= R_0_1_0_2 * dm_kl0;
            vj_ij[17] -= R_0_1_0_3 * dm_kl0;
            vj_ij[18] -= R_0_1_0_4 * dm_kl0;
            vj_ij[19] -= R_0_1_1_1 * dm_kl0;
            vj_ij[20] -= R_0_1_1_2 * dm_kl0;
            vj_ij[21] -= R_0_1_1_3 * dm_kl0;
            vj_ij[22] -= R_0_1_2_1 * dm_kl0;
            vj_ij[23] -= R_0_1_2_2 * dm_kl0;
            vj_ij[24] -= R_0_1_3_1 * dm_kl0;
            vj_ij[25] -= R_0_2_0_1 * dm_kl0;
            vj_ij[26] -= R_0_2_0_2 * dm_kl0;
            vj_ij[27] -= R_0_2_0_3 * dm_kl0;
            vj_ij[28] -= R_0_2_1_1 * dm_kl0;
            vj_ij[29] -= R_0_2_1_2 * dm_kl0;
            vj_ij[30] -= R_0_2_2_1 * dm_kl0;
            vj_ij[31] -= R_0_3_0_1 * dm_kl0;
            vj_ij[32] -= R_0_3_0_2 * dm_kl0;
            vj_ij[33] -= R_0_3_1_1 * dm_kl0;
            vj_ij[34] -= R_0_4_0_1 * dm_kl0;
            dm_kl0 = dm[kl_loc0+2];
            vj_ij[0] -= R_0_0_1_0 * dm_kl0;
            vj_ij[1] -= R_0_0_1_1 * dm_kl0;
            vj_ij[2] -= R_0_0_1_2 * dm_kl0;
            vj_ij[3] -= R_0_0_1_3 * dm_kl0;
            vj_ij[4] -= R_0_0_1_4 * dm_kl0;
            vj_ij[5] -= R_0_0_2_0 * dm_kl0;
            vj_ij[6] -= R_0_0_2_1 * dm_kl0;
            vj_ij[7] -= R_0_0_2_2 * dm_kl0;
            vj_ij[8] -= R_0_0_2_3 * dm_kl0;
            vj_ij[9] -= R_0_0_3_0 * dm_kl0;
            vj_ij[10] -= R_0_0_3_1 * dm_kl0;
            vj_ij[11] -= R_0_0_3_2 * dm_kl0;
            vj_ij[12] -= R_0_0_4_0 * dm_kl0;
            vj_ij[13] -= R_0_0_4_1 * dm_kl0;
            vj_ij[14] -= R_0_0_5_0 * dm_kl0;
            vj_ij[15] -= R_0_1_1_0 * dm_kl0;
            vj_ij[16] -= R_0_1_1_1 * dm_kl0;
            vj_ij[17] -= R_0_1_1_2 * dm_kl0;
            vj_ij[18] -= R_0_1_1_3 * dm_kl0;
            vj_ij[19] -= R_0_1_2_0 * dm_kl0;
            vj_ij[20] -= R_0_1_2_1 * dm_kl0;
            vj_ij[21] -= R_0_1_2_2 * dm_kl0;
            vj_ij[22] -= R_0_1_3_0 * dm_kl0;
            vj_ij[23] -= R_0_1_3_1 * dm_kl0;
            vj_ij[24] -= R_0_1_4_0 * dm_kl0;
            vj_ij[25] -= R_0_2_1_0 * dm_kl0;
            vj_ij[26] -= R_0_2_1_1 * dm_kl0;
            vj_ij[27] -= R_0_2_1_2 * dm_kl0;
            vj_ij[28] -= R_0_2_2_0 * dm_kl0;
            vj_ij[29] -= R_0_2_2_1 * dm_kl0;
            vj_ij[30] -= R_0_2_3_0 * dm_kl0;
            vj_ij[31] -= R_0_3_1_0 * dm_kl0;
            vj_ij[32] -= R_0_3_1_1 * dm_kl0;
            vj_ij[33] -= R_0_3_2_0 * dm_kl0;
            vj_ij[34] -= R_0_4_1_0 * dm_kl0;
            dm_kl0 = dm[kl_loc0+3];
            vj_ij[0] -= R_0_1_0_0 * dm_kl0;
            vj_ij[1] -= R_0_1_0_1 * dm_kl0;
            vj_ij[2] -= R_0_1_0_2 * dm_kl0;
            vj_ij[3] -= R_0_1_0_3 * dm_kl0;
            vj_ij[4] -= R_0_1_0_4 * dm_kl0;
            vj_ij[5] -= R_0_1_1_0 * dm_kl0;
            vj_ij[6] -= R_0_1_1_1 * dm_kl0;
            vj_ij[7] -= R_0_1_1_2 * dm_kl0;
            vj_ij[8] -= R_0_1_1_3 * dm_kl0;
            vj_ij[9] -= R_0_1_2_0 * dm_kl0;
            vj_ij[10] -= R_0_1_2_1 * dm_kl0;
            vj_ij[11] -= R_0_1_2_2 * dm_kl0;
            vj_ij[12] -= R_0_1_3_0 * dm_kl0;
            vj_ij[13] -= R_0_1_3_1 * dm_kl0;
            vj_ij[14] -= R_0_1_4_0 * dm_kl0;
            vj_ij[15] -= R_0_2_0_0 * dm_kl0;
            vj_ij[16] -= R_0_2_0_1 * dm_kl0;
            vj_ij[17] -= R_0_2_0_2 * dm_kl0;
            vj_ij[18] -= R_0_2_0_3 * dm_kl0;
            vj_ij[19] -= R_0_2_1_0 * dm_kl0;
            vj_ij[20] -= R_0_2_1_1 * dm_kl0;
            vj_ij[21] -= R_0_2_1_2 * dm_kl0;
            vj_ij[22] -= R_0_2_2_0 * dm_kl0;
            vj_ij[23] -= R_0_2_2_1 * dm_kl0;
            vj_ij[24] -= R_0_2_3_0 * dm_kl0;
            vj_ij[25] -= R_0_3_0_0 * dm_kl0;
            vj_ij[26] -= R_0_3_0_1 * dm_kl0;
            vj_ij[27] -= R_0_3_0_2 * dm_kl0;
            vj_ij[28] -= R_0_3_1_0 * dm_kl0;
            vj_ij[29] -= R_0_3_1_1 * dm_kl0;
            vj_ij[30] -= R_0_3_2_0 * dm_kl0;
            vj_ij[31] -= R_0_4_0_0 * dm_kl0;
            vj_ij[32] -= R_0_4_0_1 * dm_kl0;
            vj_ij[33] -= R_0_4_1_0 * dm_kl0;
            vj_ij[34] -= R_0_5_0_0 * dm_kl0;
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 35; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 124; n += 16) {
        int batch_kl = n % 31;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 496 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 31;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*496]);
        }
    }
}

// TILEX=48, TILEY=45
__global__ static
void md_j_5_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 720;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int sq_id = tx + 16 * ty;
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *dm = jk.dm;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 720;
    double *Rp_cache = Rq_cache + 2880;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 896;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3664; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 720; n += 256) {
        int task_kl = blockIdx.y * 720 + n;
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
            Rq_cache[n+720] = ykl;
            Rq_cache[n+1440] = zkl;
            Rq_cache[n+2160] = akl;
        } else {
            Rq_cache[n+2160] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            continue;
        }
        __syncthreads();
        if (thread_id < 16) {
            int task_ij = task_ij0 + thread_id;
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
                Rp_cache[thread_id+0] = xij;
                Rp_cache[thread_id+16] = yij;
                Rp_cache[thread_id+32] = zij;
                Rp_cache[thread_id+48] = aij;
            } else {
                Rp_cache[thread_id+48] = 1.; // aij
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
        for (int n = ty; n < 56; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[56];
        for (int ij = 0; ij < 56; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 45; ++batch_kl) {
            int task_kl0 = blockIdx.y * 720 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*45] + q_cond[pair_ij0] < bounds.cutoff) {
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
            int kl_loc0 = pair_kl_loc[task_kl];
            int ksh = pair_kl / nbas;
            int lsh = pair_kl % nbas;
            if (ksh == lsh) fac *= .5;
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+720];
            double zkl = Rq_cache[sq_kl+1440];
            double akl = Rq_cache[sq_kl+2160];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double theta_rr = theta * rr;
            eval_gamma_inc_fn(gamma_inc, theta_rr, 5, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 5; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            {
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[tx+64];
            double R_4_0_0_1 = zpq * gamma_inc[sq_id+5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl0 += R_0_0_0_5 * dm_ij_cache[tx+80];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[tx+96];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[tx+112];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[tx+128];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[tx+144];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl0 += R_0_0_1_4 * dm_ij_cache[tx+160];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[tx+176];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[tx+192];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[tx+208];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[tx+224];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[tx+240];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[tx+256];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[tx+272];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[tx+288];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl0 += R_0_0_4_1 * dm_ij_cache[tx+304];
            double R_4_0_1_0 = ypq * gamma_inc[sq_id+5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl0 += R_0_0_5_0 * dm_ij_cache[tx+320];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[tx+336];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[tx+352];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[tx+368];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[tx+384];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl0 += R_0_1_0_4 * dm_ij_cache[tx+400];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[tx+416];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[tx+432];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[tx+448];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[tx+464];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[tx+480];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[tx+496];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[tx+512];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[tx+528];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[tx+544];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl0 += R_0_1_4_0 * dm_ij_cache[tx+560];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[tx+576];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[tx+592];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[tx+608];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[tx+624];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[tx+640];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[tx+656];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[tx+672];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[tx+688];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[tx+704];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[tx+720];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[tx+736];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[tx+752];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[tx+768];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[tx+784];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[tx+800];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[tx+816];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[tx+832];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl0 += R_0_4_0_1 * dm_ij_cache[tx+848];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl0 += R_0_4_1_0 * dm_ij_cache[tx+864];
            double R_4_1_0_0 = xpq * gamma_inc[sq_id+5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl0 += R_0_5_0_0 * dm_ij_cache[tx+880];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            }{
            dm_kl0 = dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl0;
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl0;
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[4] += R_0_0_0_4 * dm_kl0;
            double R_4_0_0_1 = zpq * gamma_inc[sq_id+5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_ij[5] += R_0_0_0_5 * dm_kl0;
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_ij[6] += R_0_0_1_0 * dm_kl0;
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[7] += R_0_0_1_1 * dm_kl0;
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[8] += R_0_0_1_2 * dm_kl0;
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[9] += R_0_0_1_3 * dm_kl0;
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_ij[10] += R_0_0_1_4 * dm_kl0;
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[11] += R_0_0_2_0 * dm_kl0;
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[12] += R_0_0_2_1 * dm_kl0;
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[13] += R_0_0_2_2 * dm_kl0;
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_ij[14] += R_0_0_2_3 * dm_kl0;
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[15] += R_0_0_3_0 * dm_kl0;
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[16] += R_0_0_3_1 * dm_kl0;
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_ij[17] += R_0_0_3_2 * dm_kl0;
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[18] += R_0_0_4_0 * dm_kl0;
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_ij[19] += R_0_0_4_1 * dm_kl0;
            double R_4_0_1_0 = ypq * gamma_inc[sq_id+5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_ij[20] += R_0_0_5_0 * dm_kl0;
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_ij[21] += R_0_1_0_0 * dm_kl0;
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[22] += R_0_1_0_1 * dm_kl0;
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[23] += R_0_1_0_2 * dm_kl0;
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[24] += R_0_1_0_3 * dm_kl0;
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_ij[25] += R_0_1_0_4 * dm_kl0;
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[26] += R_0_1_1_0 * dm_kl0;
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[27] += R_0_1_1_1 * dm_kl0;
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[28] += R_0_1_1_2 * dm_kl0;
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_ij[29] += R_0_1_1_3 * dm_kl0;
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[30] += R_0_1_2_0 * dm_kl0;
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[31] += R_0_1_2_1 * dm_kl0;
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_ij[32] += R_0_1_2_2 * dm_kl0;
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[33] += R_0_1_3_0 * dm_kl0;
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_ij[34] += R_0_1_3_1 * dm_kl0;
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_ij[35] += R_0_1_4_0 * dm_kl0;
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[36] += R_0_2_0_0 * dm_kl0;
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[37] += R_0_2_0_1 * dm_kl0;
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[38] += R_0_2_0_2 * dm_kl0;
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_ij[39] += R_0_2_0_3 * dm_kl0;
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[40] += R_0_2_1_0 * dm_kl0;
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[41] += R_0_2_1_1 * dm_kl0;
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_ij[42] += R_0_2_1_2 * dm_kl0;
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[43] += R_0_2_2_0 * dm_kl0;
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_ij[44] += R_0_2_2_1 * dm_kl0;
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_ij[45] += R_0_2_3_0 * dm_kl0;
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[46] += R_0_3_0_0 * dm_kl0;
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[47] += R_0_3_0_1 * dm_kl0;
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_ij[48] += R_0_3_0_2 * dm_kl0;
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[49] += R_0_3_1_0 * dm_kl0;
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_ij[50] += R_0_3_1_1 * dm_kl0;
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_ij[51] += R_0_3_2_0 * dm_kl0;
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[52] += R_0_4_0_0 * dm_kl0;
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_ij[53] += R_0_4_0_1 * dm_kl0;
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_ij[54] += R_0_4_1_0 * dm_kl0;
            double R_4_1_0_0 = xpq * gamma_inc[sq_id+5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[sq_id+4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_ij[55] += R_0_5_0_0 * dm_kl0;
            }
        }
        double *vj_cache = Rp_cache;
#pragma unroll
        for (int n = 0; n < 56; ++n) {
            __syncthreads();
            vj_cache[thread_id] = vj_ij[n];
            for (int stride = 8; stride > 0; stride /= 2) {
                __syncthreads();
                if (ty < stride) {
                    vj_cache[thread_id] += vj_cache[thread_id + stride*16];
                }
            }
            __syncthreads();
            if (ty == 0 && task_ij0+tx < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
    }
    for (int n = tx; n < 45; n += 16) {
        int batch_kl = n % 45;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 720 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            int kl = n / 45;
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*720]);
        }
    }
}

int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds)
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
    case 0: { // lij=0, lkl=0, tilex=34, tiley=34
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 543) / 544, (npairs_kl + 543) / 544, 1);
        md_j_0_0<<<blocks, threads, 3056*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 9: { // lij=1, lkl=0, tilex=48, tiley=30
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 479) / 480, 1);
        md_j_1_0<<<blocks, threads, 3040*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 10: { // lij=1, lkl=1, tilex=17, tiley=17
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 271) / 272, (npairs_kl + 271) / 272, 1);
        md_j_1_1<<<blocks, threads, 3072*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 18: { // lij=2, lkl=0, tilex=48, tiley=26
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 415) / 416, 1);
        md_j_2_0<<<blocks, threads, 3072*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 19: { // lij=2, lkl=1, tilex=48, tiley=14
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 223) / 224, 1);
        md_j_2_1<<<blocks, threads, 3040*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 20: { // lij=2, lkl=2, tilex=20, tiley=20
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 319) / 320, (npairs_kl + 319) / 320, 1);
        md_j_2_2<<<blocks, threads, 5984*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 27: { // lij=3, lkl=0, tilex=48, tiley=20
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 319) / 320, 1);
        md_j_3_0<<<blocks, threads, 3008*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 28: { // lij=3, lkl=1, tilex=48, tiley=35
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 559) / 560, 1);
        md_j_3_1<<<blocks, threads, 6144*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 29: { // lij=3, lkl=2, tilex=48, tiley=18
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 287) / 288, 1);
        md_j_3_2<<<blocks, threads, 5952*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 36: { // lij=4, lkl=0, tilex=48, tiley=48
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 767) / 768, 1);
        md_j_4_0<<<blocks, threads, 5744*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 37: { // lij=4, lkl=1, tilex=48, tiley=31
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 495) / 496, 1);
        md_j_4_1<<<blocks, threads, 6128*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 45: { // lij=5, lkl=0, tilex=48, tiley=45
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 719) / 720, 1);
        md_j_5_0<<<blocks, threads, 6096*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    default: return 0;
    }
    return 1;
}
