#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/boys.cu"
#include "gvhf-md/md_j.cuh"


// TILEX=30, TILEY=30
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_0_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 480;
    int task_kl0 = blockIdx.y * 480;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+480 <= task_kl0) {
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
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 80 + sq_id;
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+480] = 2e5;
            Rq_cache[n+960] = 2e5;
            Rq_cache[n+1440] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 30; ++batch_ij) {
        int task_ij0 = blockIdx.x * 480 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 1; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[1];
        for (int ij = 0; ij < 1; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 30; ++batch_kl) {
            int task_kl0 = blockIdx.y * 480 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*30] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*30] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
            if (pair_ij_mapping == pair_kl_mapping) {
                if (task_ij == task_kl) fac *= .5;
                else if (task_ij < task_kl) fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 0, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
            }
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 30; n += 16) {
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

// TILEX=48, TILEY=23
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
    int task_kl0 = blockIdx.y * 368;
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
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 368;
    double *Rp_cache = Rq_cache + 1472;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 128 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 1904; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 368; n += 256) {
        int task_kl = blockIdx.y * 368 + n;
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
            Rq_cache[n+368] = ykl;
            Rq_cache[n+736] = zkl;
            Rq_cache[n+1104] = akl;
        } else {
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+368] = 2e5;
            Rq_cache[n+736] = 2e5;
            Rq_cache[n+1104] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[4];
        for (int ij = 0; ij < 4; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 23; ++batch_kl) {
            int task_kl0 = blockIdx.y * 368 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*23] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+368];
            double zkl = Rq_cache[sq_kl+736];
            double akl = Rq_cache[sq_kl+1104];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 1, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[32];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_1_0 * dm_kl0;
            vj_ij[3] += R_0_1_0_0 * dm_kl0;
            }
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 23; n += 16) {
        int kl = n / 23;
        int batch_kl = n - kl * 23;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 368 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*368]);
        }
    }
}

// TILEX=10, TILEY=10
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_1_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 160;
    int task_kl0 = blockIdx.y * 160;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+160 <= task_kl0) {
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
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 640;
    double *Rp_cache = Rq_cache + 640;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 128 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 1344; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 160; n += 256) {
        int task_kl = blockIdx.y * 160 + n;
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
            Rq_cache[n+160] = ykl;
            Rq_cache[n+320] = zkl;
            Rq_cache[n+480] = akl;
        } else {
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+160] = 2e5;
            Rq_cache[n+320] = 2e5;
            Rq_cache[n+480] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 10; ++batch_ij) {
        int task_ij0 = blockIdx.x * 160 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 4; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[4];
        for (int ij = 0; ij < 4; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 10; ++batch_kl) {
            int task_kl0 = blockIdx.y * 160 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*10] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*10] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
            if (pair_ij_mapping == pair_kl_mapping) {
                if (task_ij == task_kl) fac *= .5;
                else if (task_ij < task_kl) fac = 0.;
            }
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+160];
            double zkl = Rq_cache[sq_kl+320];
            double akl = Rq_cache[sq_kl+480];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 2, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[32];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[0];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[16];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[32];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+160] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[16];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[32];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+320] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[32];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+480] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_1_0 * dm_kl0;
            vj_ij[3] += R_0_1_0_0 * dm_kl0;
            dm_kl0 = dm[1];
            vj_ij[0] -= R_0_0_0_1 * dm_kl0;
            vj_ij[1] -= R_0_0_0_2 * dm_kl0;
            vj_ij[2] -= R_0_0_1_1 * dm_kl0;
            vj_ij[3] -= R_0_1_0_1 * dm_kl0;
            dm_kl0 = dm[2];
            vj_ij[0] -= R_0_0_1_0 * dm_kl0;
            vj_ij[1] -= R_0_0_1_1 * dm_kl0;
            vj_ij[2] -= R_0_0_2_0 * dm_kl0;
            vj_ij[3] -= R_0_1_1_0 * dm_kl0;
            dm_kl0 = dm[3];
            vj_ij[0] -= R_0_1_0_0 * dm_kl0;
            vj_ij[1] -= R_0_1_0_1 * dm_kl0;
            vj_ij[2] -= R_0_1_1_0 * dm_kl0;
            vj_ij[3] -= R_0_2_0_0 * dm_kl0;
            }
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 40; n += 16) {
        int kl = n / 10;
        int batch_kl = n - kl * 10;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 160 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*160]);
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
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 224 + sq_id;
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+256] = 2e5;
            Rq_cache[n+512] = 2e5;
            Rq_cache[n+768] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[10];
        for (int ij = 0; ij < 10; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 16; ++batch_kl) {
            int task_kl0 = blockIdx.y * 256 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*16] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 2, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[64];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[80];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[128];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
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
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 224 + sq_id;
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+480] = 2e5;
            Rq_cache[n+960] = 2e5;
            Rq_cache[n+1440] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[10];
        for (int ij = 0; ij < 10; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 30; ++batch_kl) {
            int task_kl0 = blockIdx.y * 480 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*30] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 3, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[64];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[80];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[128];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[16];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+480] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[64];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+960] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[128];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1440] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_1_0 * dm_kl0;
            vj_ij[4] += R_0_0_1_1 * dm_kl0;
            vj_ij[5] += R_0_0_2_0 * dm_kl0;
            vj_ij[6] += R_0_1_0_0 * dm_kl0;
            vj_ij[7] += R_0_1_0_1 * dm_kl0;
            vj_ij[8] += R_0_1_1_0 * dm_kl0;
            vj_ij[9] += R_0_2_0_0 * dm_kl0;
            dm_kl0 = dm[1];
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
            dm_kl0 = dm[2];
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
            dm_kl0 = dm[3];
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
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

// TILEX=14, TILEY=14
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128) static
#else
__global__ static
#endif
void md_j_2_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 224;
    int task_kl0 = blockIdx.y * 224;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+224 <= task_kl0) {
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
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2240;
    double *Rp_cache = Rq_cache + 896;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 224 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3200; n += 256) {
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+224] = 2e5;
            Rq_cache[n+448] = 2e5;
            Rq_cache[n+672] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 14; ++batch_ij) {
        int task_ij0 = blockIdx.x * 224 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 10; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[10];
        for (int ij = 0; ij < 10; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 14; ++batch_kl) {
            int task_kl0 = blockIdx.y * 224 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*14] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*14] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
            if (pair_ij_mapping == pair_kl_mapping) {
                if (task_ij == task_kl) fac *= .5;
                else if (task_ij < task_kl) fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[64];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[80];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[128];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[16];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+224] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[0];
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[16];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[32];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[48];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[64];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[80];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[96];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[112];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[128];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+448] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[64];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+672] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[0];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[16];
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[32];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[48];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[64];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[80];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[96];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[112];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[128];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+896] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[0];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[16];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[32];
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[48];
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[64];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[80];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[96];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[112];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[128];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1120] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[128];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1344] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[0];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[16];
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[32];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[48];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[64];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[80];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[96];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[112];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[128];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1568] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[0];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[16];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[32];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[48];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[64];
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[80];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[96];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[112];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[128];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1792] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[0];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[16];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[32];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[48];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[64];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[80];
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[96];
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[112];
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[128];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+2016] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            vj_ij[3] += R_0_0_1_0 * dm_kl0;
            vj_ij[4] += R_0_0_1_1 * dm_kl0;
            vj_ij[5] += R_0_0_2_0 * dm_kl0;
            vj_ij[6] += R_0_1_0_0 * dm_kl0;
            vj_ij[7] += R_0_1_0_1 * dm_kl0;
            vj_ij[8] += R_0_1_1_0 * dm_kl0;
            vj_ij[9] += R_0_2_0_0 * dm_kl0;
            dm_kl0 = dm[1];
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
            dm_kl0 = dm[2];
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
            dm_kl0 = dm[3];
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
            dm_kl0 = dm[4];
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
            dm_kl0 = dm[5];
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
            dm_kl0 = dm[6];
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
            dm_kl0 = dm[7];
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
            dm_kl0 = dm[8];
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
            dm_kl0 = dm[9];
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 140; n += 16) {
        int kl = n / 14;
        int batch_kl = n - kl * 14;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 224 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*224]);
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
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 384 + sq_id;
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+736] = 2e5;
            Rq_cache[n+1472] = 2e5;
            Rq_cache[n+2208] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[20];
        for (int ij = 0; ij < 20; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 46; ++batch_kl) {
            int task_kl0 = blockIdx.y * 736 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*46] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 3, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[48];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[96];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[128];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[144];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[240];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[288];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
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

// TILEX=48, TILEY=24
__global__ static
void md_j_3_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
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
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1536;
    double *Rp_cache = Rq_cache + 1536;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 384 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 3136; n += 256) {
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+384] = 2e5;
            Rq_cache[n+768] = 2e5;
            Rq_cache[n+1152] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[20];
        for (int ij = 0; ij < 20; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 24; ++batch_kl) {
            int task_kl0 = blockIdx.y * 384 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*24] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[48];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[96];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[128];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[144];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[240];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[288];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[32];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 -= R_0_0_0_4 * dm_ij_cache[48];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[64];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[80];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[96];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[112];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[128];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[144];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[160];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[176];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[192];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[208];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[224];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[240];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[256];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[272];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+384] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[64];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[96];
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[112];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[128];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 -= R_0_0_4_0 * dm_ij_cache[144];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[160];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[176];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[192];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[208];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[224];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[240];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[256];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[272];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+768] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[64];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[80];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[96];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[112];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[128];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[144];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[160];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[176];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[192];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[208];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[224];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[240];
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[256];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[272];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[288];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 -= R_0_4_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1152] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
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
            dm_kl0 = dm[1];
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
            dm_kl0 = dm[2];
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
            dm_kl0 = dm[3];
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 96; n += 16) {
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

// TILEX=48, TILEY=11
__global__ static
void md_j_3_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1760;
    double *Rp_cache = Rq_cache + 704;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 384 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2528; n += 256) {
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+176] = 2e5;
            Rq_cache[n+352] = 2e5;
            Rq_cache[n+528] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 20; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[20];
        for (int ij = 0; ij < 20; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 11; ++batch_kl) {
            int task_kl0 = blockIdx.y * 176 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*11] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[48];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[96];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[128];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[144];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[240];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[288];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[32];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 -= R_0_0_0_4 * dm_ij_cache[48];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[64];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[80];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[96];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[112];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[128];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[144];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[160];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[176];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[192];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[208];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[224];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[240];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[256];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[272];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+176] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[0];
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[16];
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[32];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl0 += R_0_0_0_5 * dm_ij_cache[48];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[64];
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[80];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl0 += R_0_0_1_4 * dm_ij_cache[96];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[112];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[128];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[144];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[160];
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[176];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl0 += R_0_1_0_4 * dm_ij_cache[192];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[208];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[224];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[240];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[256];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[272];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[288];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+352] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[64];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[96];
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[112];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[128];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 -= R_0_0_4_0 * dm_ij_cache[144];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[160];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[176];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[192];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[208];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[224];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[240];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[256];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[272];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+528] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[0];
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[16];
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[32];
            vj_kl0 += R_0_0_1_4 * dm_ij_cache[48];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[64];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[80];
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[96];
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[112];
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[128];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl0 += R_0_0_4_1 * dm_ij_cache[144];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[160];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[176];
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[192];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[208];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[224];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[240];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[256];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[272];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[288];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+704] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[0];
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[16];
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[32];
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[48];
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[64];
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[80];
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[96];
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[112];
            vj_kl0 += R_0_0_4_1 * dm_ij_cache[128];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl0 += R_0_0_5_0 * dm_ij_cache[144];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[160];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[176];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[192];
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[208];
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[224];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl0 += R_0_1_4_0 * dm_ij_cache[240];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[256];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[272];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[288];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+880] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[64];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[80];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[96];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[112];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[128];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[144];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[160];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[176];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[192];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[208];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[224];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[240];
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[256];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[272];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[288];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 -= R_0_4_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1056] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[0];
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[16];
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[32];
            vj_kl0 += R_0_1_0_4 * dm_ij_cache[48];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[64];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[80];
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[96];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[112];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[128];
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[144];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[160];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[176];
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[192];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[208];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[224];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[240];
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[256];
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[272];
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[288];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl0 += R_0_4_0_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1232] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[0];
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[16];
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[32];
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[48];
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[64];
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[80];
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[96];
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[112];
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[128];
            vj_kl0 += R_0_1_4_0 * dm_ij_cache[144];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[160];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[176];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[192];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[208];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[224];
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[240];
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[256];
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[272];
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[288];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl0 += R_0_4_1_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1408] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[0];
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[16];
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[32];
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[48];
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[64];
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[80];
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[96];
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[112];
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[128];
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[144];
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[160];
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[176];
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[192];
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[208];
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[224];
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[240];
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[256];
            vj_kl0 += R_0_4_0_1 * dm_ij_cache[272];
            vj_kl0 += R_0_4_1_0 * dm_ij_cache[288];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl0 += R_0_5_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+1584] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
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
            dm_kl0 = dm[1];
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
            dm_kl0 = dm[2];
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
            dm_kl0 = dm[3];
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
            dm_kl0 = dm[4];
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
            dm_kl0 = dm[5];
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
            dm_kl0 = dm[6];
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
            dm_kl0 = dm[7];
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
            dm_kl0 = dm[8];
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
            dm_kl0 = dm[9];
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 110; n += 16) {
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

// TILEX=48, TILEY=36
__global__ static
void md_j_4_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 576;
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
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 576;
    double *Rp_cache = Rq_cache + 2304;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 624 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2944; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 576; n += 256) {
        int task_kl = blockIdx.y * 576 + n;
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
            Rq_cache[n+576] = ykl;
            Rq_cache[n+1152] = zkl;
            Rq_cache[n+1728] = akl;
        } else {
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+576] = 2e5;
            Rq_cache[n+1152] = 2e5;
            Rq_cache[n+1728] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[35];
        for (int ij = 0; ij < 35; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 36; ++batch_kl) {
            int task_kl0 = blockIdx.y * 576 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*36] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+576];
            double zkl = Rq_cache[sq_kl+1152];
            double akl = Rq_cache[sq_kl+1728];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[48];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[64];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[96];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[112];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[128];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[144];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[160];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[176];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[192];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[208];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[224];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[240];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[272];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[288];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[304];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[320];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[336];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[352];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[368];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[384];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[400];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[416];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[432];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[448];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[464];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[480];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[496];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[512];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[528];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 36; n += 16) {
        int kl = n / 36;
        int batch_kl = n - kl * 36;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 576 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*576]);
        }
    }
}

// TILEX=48, TILEY=18
__global__ static
void md_j_4_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds)
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
    double *vj = jk.vj;
    double vj_kl0, dm_kl0;
    unsigned int lane_id = thread_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1152;
    double *Rp_cache = Rq_cache + 1152;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 624 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;
    for (int n = thread_id; n < 2368; n += 256) {
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+288] = 2e5;
            Rq_cache[n+576] = 2e5;
            Rq_cache[n+864] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 35; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[35];
        for (int ij = 0; ij < 35; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 18; ++batch_kl) {
            int task_kl0 = blockIdx.y * 288 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*18] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, 0, 256);
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[48];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[64];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[96];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[112];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[128];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[144];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[160];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[176];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[192];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[208];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[224];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[240];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[272];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[288];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[304];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[320];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[336];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[352];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[368];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[384];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[400];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[416];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[432];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[448];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[464];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[480];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[496];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[512];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[528];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_0_4 * dm_ij_cache[48];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl0 -= R_0_0_0_5 * dm_ij_cache[64];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[80];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[96];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[112];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl0 -= R_0_0_1_4 * dm_ij_cache[128];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[144];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[160];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl0 -= R_0_0_2_3 * dm_ij_cache[176];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[192];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl0 -= R_0_0_3_2 * dm_ij_cache[208];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl0 -= R_0_0_4_1 * dm_ij_cache[224];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[240];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[256];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[272];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl0 -= R_0_1_0_4 * dm_ij_cache[288];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[304];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[320];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl0 -= R_0_1_1_3 * dm_ij_cache[336];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[352];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl0 -= R_0_1_2_2 * dm_ij_cache[368];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl0 -= R_0_1_3_1 * dm_ij_cache[384];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[400];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[416];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl0 -= R_0_2_0_3 * dm_ij_cache[432];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[448];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl0 -= R_0_2_1_2 * dm_ij_cache[464];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl0 -= R_0_2_2_1 * dm_ij_cache[480];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[496];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl0 -= R_0_3_0_2 * dm_ij_cache[512];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl0 -= R_0_3_1_1 * dm_ij_cache[528];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl0 -= R_0_4_0_1 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+288] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl0 -= R_0_0_1_4 * dm_ij_cache[64];
            vj_kl0 -= R_0_0_2_0 * dm_ij_cache[80];
            vj_kl0 -= R_0_0_2_1 * dm_ij_cache[96];
            vj_kl0 -= R_0_0_2_2 * dm_ij_cache[112];
            vj_kl0 -= R_0_0_2_3 * dm_ij_cache[128];
            vj_kl0 -= R_0_0_3_0 * dm_ij_cache[144];
            vj_kl0 -= R_0_0_3_1 * dm_ij_cache[160];
            vj_kl0 -= R_0_0_3_2 * dm_ij_cache[176];
            vj_kl0 -= R_0_0_4_0 * dm_ij_cache[192];
            vj_kl0 -= R_0_0_4_1 * dm_ij_cache[208];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl0 -= R_0_0_5_0 * dm_ij_cache[224];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[240];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[256];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[272];
            vj_kl0 -= R_0_1_1_3 * dm_ij_cache[288];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[304];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[320];
            vj_kl0 -= R_0_1_2_2 * dm_ij_cache[336];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[352];
            vj_kl0 -= R_0_1_3_1 * dm_ij_cache[368];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl0 -= R_0_1_4_0 * dm_ij_cache[384];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[400];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[416];
            vj_kl0 -= R_0_2_1_2 * dm_ij_cache[432];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[448];
            vj_kl0 -= R_0_2_2_1 * dm_ij_cache[464];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl0 -= R_0_2_3_0 * dm_ij_cache[480];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[496];
            vj_kl0 -= R_0_3_1_1 * dm_ij_cache[512];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl0 -= R_0_3_2_0 * dm_ij_cache[528];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl0 -= R_0_4_1_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+576] += vj_kl0; }
            vj_kl0 = 0.;
            vj_kl0 -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl0 -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl0 -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl0 -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl0 -= R_0_1_0_4 * dm_ij_cache[64];
            vj_kl0 -= R_0_1_1_0 * dm_ij_cache[80];
            vj_kl0 -= R_0_1_1_1 * dm_ij_cache[96];
            vj_kl0 -= R_0_1_1_2 * dm_ij_cache[112];
            vj_kl0 -= R_0_1_1_3 * dm_ij_cache[128];
            vj_kl0 -= R_0_1_2_0 * dm_ij_cache[144];
            vj_kl0 -= R_0_1_2_1 * dm_ij_cache[160];
            vj_kl0 -= R_0_1_2_2 * dm_ij_cache[176];
            vj_kl0 -= R_0_1_3_0 * dm_ij_cache[192];
            vj_kl0 -= R_0_1_3_1 * dm_ij_cache[208];
            vj_kl0 -= R_0_1_4_0 * dm_ij_cache[224];
            vj_kl0 -= R_0_2_0_0 * dm_ij_cache[240];
            vj_kl0 -= R_0_2_0_1 * dm_ij_cache[256];
            vj_kl0 -= R_0_2_0_2 * dm_ij_cache[272];
            vj_kl0 -= R_0_2_0_3 * dm_ij_cache[288];
            vj_kl0 -= R_0_2_1_0 * dm_ij_cache[304];
            vj_kl0 -= R_0_2_1_1 * dm_ij_cache[320];
            vj_kl0 -= R_0_2_1_2 * dm_ij_cache[336];
            vj_kl0 -= R_0_2_2_0 * dm_ij_cache[352];
            vj_kl0 -= R_0_2_2_1 * dm_ij_cache[368];
            vj_kl0 -= R_0_2_3_0 * dm_ij_cache[384];
            vj_kl0 -= R_0_3_0_0 * dm_ij_cache[400];
            vj_kl0 -= R_0_3_0_1 * dm_ij_cache[416];
            vj_kl0 -= R_0_3_0_2 * dm_ij_cache[432];
            vj_kl0 -= R_0_3_1_0 * dm_ij_cache[448];
            vj_kl0 -= R_0_3_1_1 * dm_ij_cache[464];
            vj_kl0 -= R_0_3_2_0 * dm_ij_cache[480];
            vj_kl0 -= R_0_4_0_0 * dm_ij_cache[496];
            vj_kl0 -= R_0_4_0_1 * dm_ij_cache[512];
            vj_kl0 -= R_0_4_1_0 * dm_ij_cache[528];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl0 -= R_0_5_0_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+864] += vj_kl0; }
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
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
            dm_kl0 = dm[1];
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
            dm_kl0 = dm[2];
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
            dm_kl0 = dm[3];
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
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
        }
        }
    }
    for (int n = tx; n < 72; n += 16) {
        int kl = n / 18;
        int batch_kl = n - kl * 18;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 288 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*288]);
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
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 960 + sq_id;
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
            Rq_cache[n+0] = 2e5;
            Rq_cache[n+416] = 2e5;
            Rq_cache[n+832] = 2e5;
            Rq_cache[n+1248] = 1.;
        }
    }

    for (int batch_ij = 0; batch_ij < 48; ++batch_ij) {
        int task_ij0 = blockIdx.x * 768 + batch_ij * 16;
        if (task_ij0 >= npairs_ij) {
            break;
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
                Rp_cache[thread_id+0] = 2e5;
                Rp_cache[thread_id+16] = 2e5;
                Rp_cache[thread_id+32] = 2e5;
                Rp_cache[thread_id+48] = 1.; // aij
            }
        }
        int task_ij = task_ij0 + tx;
        if (task_ij >= npairs_ij) {
            task_ij = task_ij0;
        }
        int ij_loc0 = pair_ij_loc[task_ij];
        double *dm = jk.dm + ij_loc0;
        for (int n = ty; n < 56; n += 16) {
            dm_ij_cache[n*16] = dm[n];
        }
        double vj_ij[56];
        for (int ij = 0; ij < 56; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 26; ++batch_kl) {
            int task_kl0 = blockIdx.y * 416 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*26] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                fac = 0.;
            }
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, 0, 256);
            {
            vj_kl0 = 0.;
            vj_kl0 += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl0 += R_0_0_0_3 * dm_ij_cache[48];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl0 += R_0_0_0_4 * dm_ij_cache[64];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl0 += R_0_0_0_5 * dm_ij_cache[80];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl0 += R_0_0_1_0 * dm_ij_cache[96];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl0 += R_0_0_1_1 * dm_ij_cache[112];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl0 += R_0_0_1_2 * dm_ij_cache[128];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl0 += R_0_0_1_3 * dm_ij_cache[144];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl0 += R_0_0_1_4 * dm_ij_cache[160];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_0_2_0 * dm_ij_cache[176];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_0_2_1 * dm_ij_cache[192];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_0_2_2 * dm_ij_cache[208];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_0_2_3 * dm_ij_cache[224];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl0 += R_0_0_3_0 * dm_ij_cache[240];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl0 += R_0_0_3_1 * dm_ij_cache[256];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl0 += R_0_0_3_2 * dm_ij_cache[272];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl0 += R_0_0_4_0 * dm_ij_cache[288];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl0 += R_0_0_4_1 * dm_ij_cache[304];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl0 += R_0_0_5_0 * dm_ij_cache[320];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl0 += R_0_1_0_0 * dm_ij_cache[336];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl0 += R_0_1_0_1 * dm_ij_cache[352];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl0 += R_0_1_0_2 * dm_ij_cache[368];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl0 += R_0_1_0_3 * dm_ij_cache[384];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl0 += R_0_1_0_4 * dm_ij_cache[400];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl0 += R_0_1_1_0 * dm_ij_cache[416];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl0 += R_0_1_1_1 * dm_ij_cache[432];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl0 += R_0_1_1_2 * dm_ij_cache[448];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl0 += R_0_1_1_3 * dm_ij_cache[464];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl0 += R_0_1_2_0 * dm_ij_cache[480];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl0 += R_0_1_2_1 * dm_ij_cache[496];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl0 += R_0_1_2_2 * dm_ij_cache[512];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl0 += R_0_1_3_0 * dm_ij_cache[528];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl0 += R_0_1_3_1 * dm_ij_cache[544];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl0 += R_0_1_4_0 * dm_ij_cache[560];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl0 += R_0_2_0_0 * dm_ij_cache[576];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl0 += R_0_2_0_1 * dm_ij_cache[592];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl0 += R_0_2_0_2 * dm_ij_cache[608];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl0 += R_0_2_0_3 * dm_ij_cache[624];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl0 += R_0_2_1_0 * dm_ij_cache[640];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl0 += R_0_2_1_1 * dm_ij_cache[656];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl0 += R_0_2_1_2 * dm_ij_cache[672];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl0 += R_0_2_2_0 * dm_ij_cache[688];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl0 += R_0_2_2_1 * dm_ij_cache[704];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl0 += R_0_2_3_0 * dm_ij_cache[720];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl0 += R_0_3_0_0 * dm_ij_cache[736];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl0 += R_0_3_0_1 * dm_ij_cache[752];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl0 += R_0_3_0_2 * dm_ij_cache[768];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl0 += R_0_3_1_0 * dm_ij_cache[784];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl0 += R_0_3_1_1 * dm_ij_cache[800];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl0 += R_0_3_2_0 * dm_ij_cache[816];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl0 += R_0_4_0_0 * dm_ij_cache[832];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl0 += R_0_4_0_1 * dm_ij_cache[848];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl0 += R_0_4_1_0 * dm_ij_cache[864];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl0 += R_0_5_0_0 * dm_ij_cache[880];
            for (int offset = 8; offset > 0; offset /= 2) {
                vj_kl0 += __shfl_down_sync(mask, vj_kl0, offset);
            }
            if (tx == 0) { vj_kl_cache[sq_kl+0] += vj_kl0; }
            }{
            if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            double *dm = jk.dm + kl_loc0;
            dm_kl0 = dm[0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl0;
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl0;
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl0;
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl0;
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[4] += R_0_0_0_4 * dm_kl0;
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_ij[5] += R_0_0_0_5 * dm_kl0;
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[6] += R_0_0_1_0 * dm_kl0;
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[7] += R_0_0_1_1 * dm_kl0;
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[8] += R_0_0_1_2 * dm_kl0;
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[9] += R_0_0_1_3 * dm_kl0;
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_ij[10] += R_0_0_1_4 * dm_kl0;
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
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
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
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
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[18] += R_0_0_4_0 * dm_kl0;
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_ij[19] += R_0_0_4_1 * dm_kl0;
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_ij[20] += R_0_0_5_0 * dm_kl0;
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
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
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
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
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
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
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
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
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_ij[55] += R_0_5_0_0 * dm_kl0;
            }
            }
        }
        {
        double *vj_cache = Rp_cache;
        int task_ij = task_ij0 + tx;
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
            if (ty == 0 && task_ij < npairs_ij) {
                atomicAdd(vj+ij_loc0+n, vj_cache[thread_id]);
            }
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
    case 0: { // lij=0, lkl=0, tilex=30, tiley=30
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 479) / 480, (npairs_kl + 479) / 480, 1);
        md_j_0_0<<<blocks, threads, (2992+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 11: { // lij=1, lkl=0, tilex=48, tiley=23
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 367) / 368, 1);
        md_j_1_0<<<blocks, threads, (2992+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 12: { // lij=1, lkl=1, tilex=10, tiley=10
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 159) / 160, (npairs_kl + 159) / 160, 1);
        md_j_1_1<<<blocks, threads, (2944+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
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
    case 24: { // lij=2, lkl=2, tilex=14, tiley=14
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 223) / 224, (npairs_kl + 223) / 224, 1);
        cudaFuncSetAttribute(md_j_2_2, cudaFuncAttributeMaxDynamicSharedMemorySize, (5920+addition_buf)*sizeof(double));
        md_j_2_2<<<blocks, threads, (5920+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 33: { // lij=3, lkl=0, tilex=48, tiley=46
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 735) / 736, 1);
        cudaFuncSetAttribute(md_j_3_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6112+addition_buf)*sizeof(double));
        md_j_3_0<<<blocks, threads, (6112+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 34: { // lij=3, lkl=1, tilex=48, tiley=24
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 383) / 384, 1);
        cudaFuncSetAttribute(md_j_3_1, cudaFuncAttributeMaxDynamicSharedMemorySize, (6016+addition_buf)*sizeof(double));
        md_j_3_1<<<blocks, threads, (6016+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 35: { // lij=3, lkl=2, tilex=48, tiley=11
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 175) / 176, 1);
        cudaFuncSetAttribute(md_j_3_2, cudaFuncAttributeMaxDynamicSharedMemorySize, (5920+addition_buf)*sizeof(double));
        md_j_3_2<<<blocks, threads, (5920+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 44: { // lij=4, lkl=0, tilex=48, tiley=36
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 575) / 576, 1);
        cudaFuncSetAttribute(md_j_4_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6064+addition_buf)*sizeof(double));
        md_j_4_0<<<blocks, threads, (6064+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
    } break;
    case 45: { // lij=4, lkl=1, tilex=48, tiley=18
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 287) / 288, 1);
        cudaFuncSetAttribute(md_j_4_1, cudaFuncAttributeMaxDynamicSharedMemorySize, (6000+addition_buf)*sizeof(double));
        md_j_4_1<<<blocks, threads, (6000+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds);
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
