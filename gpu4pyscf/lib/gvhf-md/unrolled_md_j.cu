#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf1.cuh"
#include "gvhf-md/boys.cu"
#include "gvhf-md/md_j.cuh"


// TILEX=31, TILEY=31
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_0_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 496;
    int task_kl0 = blockIdx.y * 496;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+496 <= task_kl0) {
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
    double *Rq_cache = vj_kl_cache + 496;
    double *Rp_cache = Rq_cache + 1984;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 16;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2544; n += 256) {
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

    for (int batch_ij = 0; batch_ij < 31; ++batch_ij) {
        int task_ij0 = blockIdx.x * 496 + batch_ij * 16;
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
        for (int batch_kl = 0; batch_kl < 31; ++batch_kl) {
            int task_kl0 = blockIdx.y * 496 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*31] + q_cond[pair_kl0] < bounds.cutoff &&
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
            double ykl = Rq_cache[sq_kl+496];
            double zkl = Rq_cache[sq_kl+992];
            double akl = Rq_cache[sq_kl+1488];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 0, sq_id, 256);
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
    for (int n = tx; n < 31; n += 16) {
        int kl = n / 31;
        int batch_kl = n - kl * 31;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 496 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*496]);
        }
    }
}

// TILEX=48, TILEY=24
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
    int task_kl0 = blockIdx.y * 384;
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
    double *Rq_cache = vj_kl_cache + 384;
    double *Rp_cache = Rq_cache + 1536;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 1984; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 384; n += 256) {
        int task_kl = blockIdx.y * 384 + n;
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
            Rq_cache[n+384] = ykl;
            Rq_cache[n+768] = zkl;
            Rq_cache[n+1152] = akl;
        } else {
            Rq_cache[n+1152] = 1.;
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
        for (int batch_kl = 0; batch_kl < 24; ++batch_kl) {
            int task_kl0 = blockIdx.y * 384 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*24] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+384];
            double zkl = Rq_cache[sq_kl+768];
            double akl = Rq_cache[sq_kl+1152];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 1, sq_id, 256);
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
    for (int n = tx; n < 24; n += 16) {
        int kl = n / 24;
        int batch_kl = n - kl * 24;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 384 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*384]);
        }
    }
}

// TILEX=11, TILEY=11
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_1_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 176;
    int task_kl0 = blockIdx.y * 176;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+176 <= task_kl0) {
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
    double *Rq_cache = vj_kl_cache + 704;
    double *Rp_cache = Rq_cache + 704;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 64;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 1472; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 176; n += 256) {
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
        } else {
            Rq_cache[n+528] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 11; ++batch_ij) {
        int task_ij0 = blockIdx.x * 176 + batch_ij * 16;
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
        for (int batch_kl = 0; batch_kl < 11; ++batch_kl) {
            int task_kl0 = blockIdx.y * 176 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*11] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*11] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+176];
            double zkl = Rq_cache[sq_kl+352];
            double akl = Rq_cache[sq_kl+528];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 2, sq_id, 256);
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
            if (tx == 0) { vj_kl_cache[sq_kl+176] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+352] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+528] += vj_kl0; }
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
    for (int n = tx; n < 44; n += 16) {
        int kl = n / 11;
        int batch_kl = n - kl * 11;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 176 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*176]);
        }
    }
}

// TILEX=48, TILEY=16
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
    double *Rq_cache = vj_kl_cache + 256;
    double *Rp_cache = Rq_cache + 1024;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 1344; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 256; n += 256) {
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
        for (int batch_kl = 0; batch_kl < 16; ++batch_kl) {
            int task_kl0 = blockIdx.y * 256 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*16] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+256];
            double zkl = Rq_cache[sq_kl+512];
            double akl = Rq_cache[sq_kl+768];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 2, sq_id, 256);
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
    for (int n = tx; n < 16; n += 16) {
        int kl = n / 16;
        int batch_kl = n - kl * 16;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 256 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*256]);
        }
    }
}

// TILEX=48, TILEY=30
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
    double *Rq_cache = vj_kl_cache + 1920;
    double *Rp_cache = Rq_cache + 1920;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3904; n += 256) {
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
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[10];
        for (int ij = 0; ij < 10; ++ij) {
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
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 3, sq_id, 256);
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
            if (tx == 0) { vj_kl_cache[sq_kl+480] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1440] += vj_kl0; }
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
    for (int n = tx; n < 120; n += 16) {
        int kl = n / 30;
        int batch_kl = n - kl * 30;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 480 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*480]);
        }
    }
}

// TILEX=15, TILEY=15
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_2_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 240;
    int task_kl0 = blockIdx.y * 240;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+240 <= task_kl0) {
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
    double *Rq_cache = vj_kl_cache + 2400;
    double *Rp_cache = Rq_cache + 960;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 160;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3424; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 240; n += 256) {
        int task_kl = blockIdx.y * 240 + n;
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
            Rq_cache[n+240] = ykl;
            Rq_cache[n+480] = zkl;
            Rq_cache[n+720] = akl;
        } else {
            Rq_cache[n+720] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 15; ++batch_ij) {
        int task_ij0 = blockIdx.x * 240 + batch_ij * 16;
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
        for (int batch_kl = 0; batch_kl < 15; ++batch_kl) {
            int task_kl0 = blockIdx.y * 240 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*15] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*15] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+240];
            double zkl = Rq_cache[sq_kl+480];
            double akl = Rq_cache[sq_kl+720];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, sq_id, 256);
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
            if (tx == 0) { vj_kl_cache[sq_kl+240] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+480] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+720] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+960] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1200] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1440] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1680] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1920] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+2160] += vj_kl0; }
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
    for (int n = tx; n < 150; n += 16) {
        int kl = n / 15;
        int batch_kl = n - kl * 15;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 240 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*240]);
        }
    }
}

// TILEX=48, TILEY=46
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
    int task_kl0 = blockIdx.y * 736;
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
    double *Rq_cache = vj_kl_cache + 736;
    double *Rp_cache = Rq_cache + 2944;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3744; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 736; n += 256) {
        int task_kl = blockIdx.y * 736 + n;
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
            Rq_cache[n+736] = ykl;
            Rq_cache[n+1472] = zkl;
            Rq_cache[n+2208] = akl;
        } else {
            Rq_cache[n+2208] = 1.;
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
        for (int batch_kl = 0; batch_kl < 46; ++batch_kl) {
            int task_kl0 = blockIdx.y * 736 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*46] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+736];
            double zkl = Rq_cache[sq_kl+1472];
            double akl = Rq_cache[sq_kl+2208];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 3, sq_id, 256);
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
    for (int n = tx; n < 46; n += 16) {
        int kl = n / 46;
        int batch_kl = n - kl * 46;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 736 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*736]);
        }
    }
}

// TILEX=48, TILEY=25
__global__ static
void md_j_3_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    double *Rq_cache = vj_kl_cache + 1600;
    double *Rp_cache = Rq_cache + 1600;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3264; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 400; n += 256) {
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
        } else {
            Rq_cache[n+1200] = 1.;
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
        for (int batch_kl = 0; batch_kl < 25; ++batch_kl) {
            int task_kl0 = blockIdx.y * 400 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*25] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+400];
            double zkl = Rq_cache[sq_kl+800];
            double akl = Rq_cache[sq_kl+1200];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, sq_id, 256);
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
            if (tx == 0) { vj_kl_cache[sq_kl+400] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+800] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1200] += vj_kl0; }
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
    for (int n = tx; n < 100; n += 16) {
        int kl = n / 25;
        int batch_kl = n - kl * 25;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 400 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*400]);
        }
    }
}

// TILEX=48, TILEY=12
__global__ static
void md_j_3_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    double *Rq_cache = vj_kl_cache + 1920;
    double *Rp_cache = Rq_cache + 768;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 320;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2752; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 192; n += 256) {
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
        } else {
            Rq_cache[n+576] = 1.;
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
        for (int batch_kl = 0; batch_kl < 12; ++batch_kl) {
            int task_kl0 = blockIdx.y * 192 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*12] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+192];
            double zkl = Rq_cache[sq_kl+384];
            double akl = Rq_cache[sq_kl+576];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, sq_id, 256);
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
            if (tx == 0) { vj_kl_cache[sq_kl+192] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+384] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+576] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+768] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+960] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1152] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1344] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1536] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+1728] += vj_kl0; }
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
    for (int n = tx; n < 120; n += 16) {
        int kl = n / 12;
        int batch_kl = n - kl * 12;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 192 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*192]);
        }
    }
}

// TILEX=48, TILEY=37
__global__ static
void md_j_4_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 592;
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
    double *Rq_cache = vj_kl_cache + 592;
    double *Rp_cache = Rq_cache + 2368;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 560;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3024; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 592; n += 256) {
        int task_kl = blockIdx.y * 592 + n;
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
            Rq_cache[n+592] = ykl;
            Rq_cache[n+1184] = zkl;
            Rq_cache[n+1776] = akl;
        } else {
            Rq_cache[n+1776] = 1.;
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
        for (int batch_kl = 0; batch_kl < 37; ++batch_kl) {
            int task_kl0 = blockIdx.y * 592 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*37] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+592];
            double zkl = Rq_cache[sq_kl+1184];
            double akl = Rq_cache[sq_kl+1776];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, sq_id, 256);
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
    for (int n = tx; n < 37; n += 16) {
        int kl = n / 37;
        int batch_kl = n - kl * 37;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 592 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*592]);
        }
    }
}

// TILEX=48, TILEY=19
__global__ static
void md_j_4_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 304;
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
    double *Rq_cache = vj_kl_cache + 1216;
    double *Rp_cache = Rq_cache + 1216;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 560;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2496; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 304; n += 256) {
        int task_kl = blockIdx.y * 304 + n;
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
            Rq_cache[n+304] = ykl;
            Rq_cache[n+608] = zkl;
            Rq_cache[n+912] = akl;
        } else {
            Rq_cache[n+912] = 1.;
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
        for (int batch_kl = 0; batch_kl < 19; ++batch_kl) {
            int task_kl0 = blockIdx.y * 304 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*19] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+304];
            double zkl = Rq_cache[sq_kl+608];
            double akl = Rq_cache[sq_kl+912];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, sq_id, 256);
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
            if (tx == 0) { vj_kl_cache[sq_kl+304] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+608] += vj_kl0; }
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
            if (tx == 0) { vj_kl_cache[sq_kl+912] += vj_kl0; }
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
    for (int n = tx; n < 76; n += 16) {
        int kl = n / 19;
        int batch_kl = n - kl * 19;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 304 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*304]);
        }
    }
}

// TILEX=48, TILEY=26
__global__ static
void md_j_5_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
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
    double *gamma_inc = dm_ij_cache + 896;
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
        for (int n = ty; n < 56; n += 16) {
            dm_ij_cache[tx+n*16] = dm[ij_loc0+n];
        }
        double vj_ij[56];
        for (int ij = 0; ij < 56; ++ij) {
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
            double omega = env[PTR_RANGE_OMEGA];
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, sq_id, 256);
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
    for (int n = tx; n < 26; n += 16) {
        int kl = n / 26;
        int batch_kl = n - kl * 26;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 416 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*416]);
        }
    }
}

int md_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds, double omega)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int ijkl = lij*11 + lkl;
    int npairs_ij = bounds->npairs_ij;
    int npairs_kl = bounds->npairs_kl;
    int addition_buf = 0;
    if (omega < 0) {
        addition_buf = 256;
    }
    switch (ijkl) {
    case 0: { // lij=0, lkl=0, tilex=31, tiley=31
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 495) / 496, (npairs_kl + 495) / 496, 1);
        md_j_0_0<<<blocks, threads, (3072+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 11: { // lij=1, lkl=0, tilex=48, tiley=24
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 383) / 384, 1);
        md_j_1_0<<<blocks, threads, (3072+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 12: { // lij=1, lkl=1, tilex=11, tiley=11
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 175) / 176, (npairs_kl + 175) / 176, 1);
        md_j_1_1<<<blocks, threads, (3072+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 22: { // lij=2, lkl=0, tilex=48, tiley=16
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 255) / 256, 1);
        md_j_2_0<<<blocks, threads, (3040+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 23: { // lij=2, lkl=1, tilex=48, tiley=30
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 479) / 480, 1);
        cudaFuncSetAttribute(md_j_2_1, cudaFuncAttributeMaxDynamicSharedMemorySize, (6112+addition_buf)*sizeof(double));
        md_j_2_1<<<blocks, threads, (6112+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 24: { // lij=2, lkl=2, tilex=15, tiley=15
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 239) / 240, (npairs_kl + 239) / 240, 1);
        cudaFuncSetAttribute(md_j_2_2, cudaFuncAttributeMaxDynamicSharedMemorySize, (6144+addition_buf)*sizeof(double));
        md_j_2_2<<<blocks, threads, (6144+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 33: { // lij=3, lkl=0, tilex=48, tiley=46
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 735) / 736, 1);
        cudaFuncSetAttribute(md_j_3_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6112+addition_buf)*sizeof(double));
        md_j_3_0<<<blocks, threads, (6112+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 34: { // lij=3, lkl=1, tilex=48, tiley=25
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 399) / 400, 1);
        cudaFuncSetAttribute(md_j_3_1, cudaFuncAttributeMaxDynamicSharedMemorySize, (6144+addition_buf)*sizeof(double));
        md_j_3_1<<<blocks, threads, (6144+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 35: { // lij=3, lkl=2, tilex=48, tiley=12
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 191) / 192, 1);
        cudaFuncSetAttribute(md_j_3_2, cudaFuncAttributeMaxDynamicSharedMemorySize, (6144+addition_buf)*sizeof(double));
        md_j_3_2<<<blocks, threads, (6144+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 44: { // lij=4, lkl=0, tilex=48, tiley=37
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 591) / 592, 1);
        cudaFuncSetAttribute(md_j_4_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6144+addition_buf)*sizeof(double));
        md_j_4_0<<<blocks, threads, (6144+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 45: { // lij=4, lkl=1, tilex=48, tiley=19
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 303) / 304, 1);
        cudaFuncSetAttribute(md_j_4_1, cudaFuncAttributeMaxDynamicSharedMemorySize, (6128+addition_buf)*sizeof(double));
        md_j_4_1<<<blocks, threads, (6128+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 55: { // lij=5, lkl=0, tilex=48, tiley=26
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 415) / 416, 1);
        cudaFuncSetAttribute(md_j_5_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6112+addition_buf)*sizeof(double));
        md_j_5_0<<<blocks, threads, (6112+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    default: return 0;
    }
    return 1;
}
