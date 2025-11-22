#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/boys.cu"
#include "gvhf-md/md_j.cuh"


// TILEX=21, TILEY=21
__global__ static
void md_j_4dm_0_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 336;
    int task_kl0 = blockIdx.y * 336;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+336 <= task_kl0) {
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
    double vj_kl[8];
    double dm_kl[8];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2688;
    double *Rp_cache = Rq_cache + 1344;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 192 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 1408; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 336; n += 256) {
        int task_kl = blockIdx.y * 336 + n;
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
            Rq_cache[n+336] = ykl;
            Rq_cache[n+672] = zkl;
            Rq_cache[n+1008] = akl;
        } else {
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+336] = 1e5;
            Rq_cache[n+672] = 1e5;
            Rq_cache[n+1008] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 8) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 2688; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    for (int batch_ij = 0; batch_ij < 21; ++batch_ij) {
        int task_ij0 = blockIdx.x * 336 + batch_ij * 16;
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
        int nf3ij_dm = 1 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 1;
            int i = n - i_dm * 1;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[8];
        for (int ij = 0; ij < 8; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 21; ++batch_kl) {
            int task_kl0 = blockIdx.y * 336 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*21] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*21] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            double ykl = Rq_cache[sq_kl+336];
            double zkl = Rq_cache[sq_kl+672];
            double akl = Rq_cache[sq_kl+1008];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 0, 0, 256);
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[16];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += gamma_inc[0*256] * dm_kl[1];
            } else {
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[16];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[32];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[48];
            vj_kl[4] += gamma_inc[0*256] * dm_ij_cache[64];
            vj_kl[5] += gamma_inc[0*256] * dm_ij_cache[80];
            vj_kl[6] += gamma_inc[0*256] * dm_ij_cache[96];
            vj_kl[7] += gamma_inc[0*256] * dm_ij_cache[112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[2] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[3] += gamma_inc[0*256] * dm_kl[3];
            vj_ij[4] += gamma_inc[0*256] * dm_kl[4];
            vj_ij[5] += gamma_inc[0*256] * dm_kl[5];
            vj_ij[6] += gamma_inc[0*256] * dm_kl[6];
            vj_ij[7] += gamma_inc[0*256] * dm_kl[7];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 1; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+1];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 1; ++n) {
                __syncthreads();
                for (int m = 0; m < 8; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+1*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 8; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 21; n += 16) {
        int kl = n / 21;
        int batch_kl = n - kl * 21;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 336 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*336+kl*336]);
            }
        }
    }
} }

// TILEX=48, TILEY=21
__global__ static
void md_j_4dm_1_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 336;
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
    double vj_kl[8];
    double dm_kl[8];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2688;
    double *Rp_cache = Rq_cache + 1344;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 576 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 1408; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 336; n += 256) {
        int task_kl = blockIdx.y * 336 + n;
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
            Rq_cache[n+336] = ykl;
            Rq_cache[n+672] = zkl;
            Rq_cache[n+1008] = akl;
        } else {
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+336] = 1e5;
            Rq_cache[n+672] = 1e5;
            Rq_cache[n+1008] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 8) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 2688; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 4 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 4;
            int i = n - i_dm * 4;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[32];
        for (int ij = 0; ij < 32; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 21; ++batch_kl) {
            int task_kl0 = blockIdx.y * 336 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*21] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+336];
            double zkl = Rq_cache[sq_kl+672];
            double akl = Rq_cache[sq_kl+1008];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 1, 0, 256);
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[32];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[64];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[80];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[96];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[5] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[6] += R_0_0_1_0 * dm_kl[1];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_0 * dm_kl[1];
            } else {
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[64];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[128];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[192];
            vj_kl[4] += gamma_inc[0*256] * dm_ij_cache[256];
            vj_kl[5] += gamma_inc[0*256] * dm_ij_cache[320];
            vj_kl[6] += gamma_inc[0*256] * dm_ij_cache[384];
            vj_kl[7] += gamma_inc[0*256] * dm_ij_cache[448];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[80];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[144];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[208];
            vj_kl[4] += R_0_0_0_1 * dm_ij_cache[272];
            vj_kl[5] += R_0_0_0_1 * dm_ij_cache[336];
            vj_kl[6] += R_0_0_0_1 * dm_ij_cache[400];
            vj_kl[7] += R_0_0_0_1 * dm_ij_cache[464];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[96];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[160];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[224];
            vj_kl[4] += R_0_0_1_0 * dm_ij_cache[288];
            vj_kl[5] += R_0_0_1_0 * dm_ij_cache[352];
            vj_kl[6] += R_0_0_1_0 * dm_ij_cache[416];
            vj_kl[7] += R_0_0_1_0 * dm_ij_cache[480];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[112];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[176];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[240];
            vj_kl[4] += R_0_1_0_0 * dm_ij_cache[304];
            vj_kl[5] += R_0_1_0_0 * dm_ij_cache[368];
            vj_kl[6] += R_0_1_0_0 * dm_ij_cache[432];
            vj_kl[7] += R_0_1_0_0 * dm_ij_cache[496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[8] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[12] += gamma_inc[0*256] * dm_kl[3];
            vj_ij[16] += gamma_inc[0*256] * dm_kl[4];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[5];
            vj_ij[24] += gamma_inc[0*256] * dm_kl[6];
            vj_ij[28] += gamma_inc[0*256] * dm_kl[7];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[5] += R_0_0_0_1 * dm_kl[1];
            vj_ij[9] += R_0_0_0_1 * dm_kl[2];
            vj_ij[13] += R_0_0_0_1 * dm_kl[3];
            vj_ij[17] += R_0_0_0_1 * dm_kl[4];
            vj_ij[21] += R_0_0_0_1 * dm_kl[5];
            vj_ij[25] += R_0_0_0_1 * dm_kl[6];
            vj_ij[29] += R_0_0_0_1 * dm_kl[7];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[6] += R_0_0_1_0 * dm_kl[1];
            vj_ij[10] += R_0_0_1_0 * dm_kl[2];
            vj_ij[14] += R_0_0_1_0 * dm_kl[3];
            vj_ij[18] += R_0_0_1_0 * dm_kl[4];
            vj_ij[22] += R_0_0_1_0 * dm_kl[5];
            vj_ij[26] += R_0_0_1_0 * dm_kl[6];
            vj_ij[30] += R_0_0_1_0 * dm_kl[7];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_0 * dm_kl[1];
            vj_ij[11] += R_0_1_0_0 * dm_kl[2];
            vj_ij[15] += R_0_1_0_0 * dm_kl[3];
            vj_ij[19] += R_0_1_0_0 * dm_kl[4];
            vj_ij[23] += R_0_1_0_0 * dm_kl[5];
            vj_ij[27] += R_0_1_0_0 * dm_kl[6];
            vj_ij[31] += R_0_1_0_0 * dm_kl[7];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+4];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                __syncthreads();
                for (int m = 0; m < 8; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+4*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 8; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 21; n += 16) {
        int kl = n / 21;
        int batch_kl = n - kl * 21;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 336 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*336+kl*336]);
            }
        }
    }
} }

// TILEX=6, TILEY=6
__global__ static
void md_j_4dm_1_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 96;
    int task_kl0 = blockIdx.y * 96;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+96 <= task_kl0) {
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
    double vj_kl[8];
    double dm_kl[8];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 3072;
    double *Rp_cache = Rq_cache + 384;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 576 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 448; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 96; n += 256) {
        int task_kl = blockIdx.y * 96 + n;
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
            Rq_cache[n+96] = ykl;
            Rq_cache[n+192] = zkl;
            Rq_cache[n+288] = akl;
        } else {
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+96] = 1e5;
            Rq_cache[n+192] = 1e5;
            Rq_cache[n+288] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 8) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 3072; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    for (int batch_ij = 0; batch_ij < 6; ++batch_ij) {
        int task_ij0 = blockIdx.x * 96 + batch_ij * 16;
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
        int nf3ij_dm = 4 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 4;
            int i = n - i_dm * 4;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[32];
        for (int ij = 0; ij < 32; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 6; ++batch_kl) {
            int task_kl0 = blockIdx.y * 96 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*6] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*6] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            double ykl = Rq_cache[sq_kl+96];
            double zkl = Rq_cache[sq_kl+192];
            double akl = Rq_cache[sq_kl+288];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 2, 0, 256);
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[32];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[32];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+96] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[32];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+192] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[32];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+288] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+1];
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[2] += R_0_0_1_1 * dm_kl[0];
            vj_ij[3] += R_0_1_0_1 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+2];
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[2] += R_0_0_2_0 * dm_kl[0];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+3];
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[2] += R_0_1_1_0 * dm_kl[0];
            vj_ij[3] += R_0_2_0_0 * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[64];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[80];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[96];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[64];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+96] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[64];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[80];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[96];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+192] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[64];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[80];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[96];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+288] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[5] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[6] += R_0_0_1_0 * dm_kl[1];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[4] += R_0_0_0_1 * dm_kl[1];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[5] += R_0_0_0_2 * dm_kl[1];
            vj_ij[2] += R_0_0_1_1 * dm_kl[0];
            vj_ij[6] += R_0_0_1_1 * dm_kl[1];
            vj_ij[3] += R_0_1_0_1 * dm_kl[0];
            vj_ij[7] += R_0_1_0_1 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2]; }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[4] += R_0_0_1_0 * dm_kl[1];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[5] += R_0_0_1_1 * dm_kl[1];
            vj_ij[2] += R_0_0_2_0 * dm_kl[0];
            vj_ij[6] += R_0_0_2_0 * dm_kl[1];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[7] += R_0_1_1_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[4] += R_0_1_0_0 * dm_kl[1];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[5] += R_0_1_0_1 * dm_kl[1];
            vj_ij[2] += R_0_1_1_0 * dm_kl[0];
            vj_ij[6] += R_0_1_1_0 * dm_kl[1];
            vj_ij[3] += R_0_2_0_0 * dm_kl[0];
            vj_ij[7] += R_0_2_0_0 * dm_kl[1];
            } else {
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[64];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[128];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[192];
            vj_kl[4] += gamma_inc[0*256] * dm_ij_cache[256];
            vj_kl[5] += gamma_inc[0*256] * dm_ij_cache[320];
            vj_kl[6] += gamma_inc[0*256] * dm_ij_cache[384];
            vj_kl[7] += gamma_inc[0*256] * dm_ij_cache[448];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[80];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[144];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[208];
            vj_kl[4] += R_0_0_0_1 * dm_ij_cache[272];
            vj_kl[5] += R_0_0_0_1 * dm_ij_cache[336];
            vj_kl[6] += R_0_0_0_1 * dm_ij_cache[400];
            vj_kl[7] += R_0_0_0_1 * dm_ij_cache[464];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[96];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[160];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[224];
            vj_kl[4] += R_0_0_1_0 * dm_ij_cache[288];
            vj_kl[5] += R_0_0_1_0 * dm_ij_cache[352];
            vj_kl[6] += R_0_0_1_0 * dm_ij_cache[416];
            vj_kl[7] += R_0_0_1_0 * dm_ij_cache[480];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[112];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[176];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[240];
            vj_kl[4] += R_0_1_0_0 * dm_ij_cache[304];
            vj_kl[5] += R_0_1_0_0 * dm_ij_cache[368];
            vj_kl[6] += R_0_1_0_0 * dm_ij_cache[432];
            vj_kl[7] += R_0_1_0_0 * dm_ij_cache[496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[64];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[128];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[192];
            vj_kl[4] -= R_0_0_0_1 * dm_ij_cache[256];
            vj_kl[5] -= R_0_0_0_1 * dm_ij_cache[320];
            vj_kl[6] -= R_0_0_0_1 * dm_ij_cache[384];
            vj_kl[7] -= R_0_0_0_1 * dm_ij_cache[448];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[80];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[144];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[208];
            vj_kl[4] -= R_0_0_0_2 * dm_ij_cache[272];
            vj_kl[5] -= R_0_0_0_2 * dm_ij_cache[336];
            vj_kl[6] -= R_0_0_0_2 * dm_ij_cache[400];
            vj_kl[7] -= R_0_0_0_2 * dm_ij_cache[464];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[96];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[160];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[224];
            vj_kl[4] -= R_0_0_1_1 * dm_ij_cache[288];
            vj_kl[5] -= R_0_0_1_1 * dm_ij_cache[352];
            vj_kl[6] -= R_0_0_1_1 * dm_ij_cache[416];
            vj_kl[7] -= R_0_0_1_1 * dm_ij_cache[480];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[112];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[240];
            vj_kl[4] -= R_0_1_0_1 * dm_ij_cache[304];
            vj_kl[5] -= R_0_1_0_1 * dm_ij_cache[368];
            vj_kl[6] -= R_0_1_0_1 * dm_ij_cache[432];
            vj_kl[7] -= R_0_1_0_1 * dm_ij_cache[496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+96] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[64];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[128];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[192];
            vj_kl[4] -= R_0_0_1_0 * dm_ij_cache[256];
            vj_kl[5] -= R_0_0_1_0 * dm_ij_cache[320];
            vj_kl[6] -= R_0_0_1_0 * dm_ij_cache[384];
            vj_kl[7] -= R_0_0_1_0 * dm_ij_cache[448];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[80];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[144];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[208];
            vj_kl[4] -= R_0_0_1_1 * dm_ij_cache[272];
            vj_kl[5] -= R_0_0_1_1 * dm_ij_cache[336];
            vj_kl[6] -= R_0_0_1_1 * dm_ij_cache[400];
            vj_kl[7] -= R_0_0_1_1 * dm_ij_cache[464];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[96];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[160];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[224];
            vj_kl[4] -= R_0_0_2_0 * dm_ij_cache[288];
            vj_kl[5] -= R_0_0_2_0 * dm_ij_cache[352];
            vj_kl[6] -= R_0_0_2_0 * dm_ij_cache[416];
            vj_kl[7] -= R_0_0_2_0 * dm_ij_cache[480];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[112];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[176];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[240];
            vj_kl[4] -= R_0_1_1_0 * dm_ij_cache[304];
            vj_kl[5] -= R_0_1_1_0 * dm_ij_cache[368];
            vj_kl[6] -= R_0_1_1_0 * dm_ij_cache[432];
            vj_kl[7] -= R_0_1_1_0 * dm_ij_cache[496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+192] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[64];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[128];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[192];
            vj_kl[4] -= R_0_1_0_0 * dm_ij_cache[256];
            vj_kl[5] -= R_0_1_0_0 * dm_ij_cache[320];
            vj_kl[6] -= R_0_1_0_0 * dm_ij_cache[384];
            vj_kl[7] -= R_0_1_0_0 * dm_ij_cache[448];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[80];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[144];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[208];
            vj_kl[4] -= R_0_1_0_1 * dm_ij_cache[272];
            vj_kl[5] -= R_0_1_0_1 * dm_ij_cache[336];
            vj_kl[6] -= R_0_1_0_1 * dm_ij_cache[400];
            vj_kl[7] -= R_0_1_0_1 * dm_ij_cache[464];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[160];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[224];
            vj_kl[4] -= R_0_1_1_0 * dm_ij_cache[288];
            vj_kl[5] -= R_0_1_1_0 * dm_ij_cache[352];
            vj_kl[6] -= R_0_1_1_0 * dm_ij_cache[416];
            vj_kl[7] -= R_0_1_1_0 * dm_ij_cache[480];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[112];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[176];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[240];
            vj_kl[4] -= R_0_2_0_0 * dm_ij_cache[304];
            vj_kl[5] -= R_0_2_0_0 * dm_ij_cache[368];
            vj_kl[6] -= R_0_2_0_0 * dm_ij_cache[432];
            vj_kl[7] -= R_0_2_0_0 * dm_ij_cache[496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*384+288] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[8] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[12] += gamma_inc[0*256] * dm_kl[3];
            vj_ij[16] += gamma_inc[0*256] * dm_kl[4];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[5];
            vj_ij[24] += gamma_inc[0*256] * dm_kl[6];
            vj_ij[28] += gamma_inc[0*256] * dm_kl[7];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[5] += R_0_0_0_1 * dm_kl[1];
            vj_ij[9] += R_0_0_0_1 * dm_kl[2];
            vj_ij[13] += R_0_0_0_1 * dm_kl[3];
            vj_ij[17] += R_0_0_0_1 * dm_kl[4];
            vj_ij[21] += R_0_0_0_1 * dm_kl[5];
            vj_ij[25] += R_0_0_0_1 * dm_kl[6];
            vj_ij[29] += R_0_0_0_1 * dm_kl[7];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[6] += R_0_0_1_0 * dm_kl[1];
            vj_ij[10] += R_0_0_1_0 * dm_kl[2];
            vj_ij[14] += R_0_0_1_0 * dm_kl[3];
            vj_ij[18] += R_0_0_1_0 * dm_kl[4];
            vj_ij[22] += R_0_0_1_0 * dm_kl[5];
            vj_ij[26] += R_0_0_1_0 * dm_kl[6];
            vj_ij[30] += R_0_0_1_0 * dm_kl[7];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_0 * dm_kl[1];
            vj_ij[11] += R_0_1_0_0 * dm_kl[2];
            vj_ij[15] += R_0_1_0_0 * dm_kl[3];
            vj_ij[19] += R_0_1_0_0 * dm_kl[4];
            vj_ij[23] += R_0_1_0_0 * dm_kl[5];
            vj_ij[27] += R_0_1_0_0 * dm_kl[6];
            vj_ij[31] += R_0_1_0_0 * dm_kl[7];
            for (int m = 0; m < 8; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1];
            }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[4] += R_0_0_0_1 * dm_kl[1];
            vj_ij[8] += R_0_0_0_1 * dm_kl[2];
            vj_ij[12] += R_0_0_0_1 * dm_kl[3];
            vj_ij[16] += R_0_0_0_1 * dm_kl[4];
            vj_ij[20] += R_0_0_0_1 * dm_kl[5];
            vj_ij[24] += R_0_0_0_1 * dm_kl[6];
            vj_ij[28] += R_0_0_0_1 * dm_kl[7];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[5] += R_0_0_0_2 * dm_kl[1];
            vj_ij[9] += R_0_0_0_2 * dm_kl[2];
            vj_ij[13] += R_0_0_0_2 * dm_kl[3];
            vj_ij[17] += R_0_0_0_2 * dm_kl[4];
            vj_ij[21] += R_0_0_0_2 * dm_kl[5];
            vj_ij[25] += R_0_0_0_2 * dm_kl[6];
            vj_ij[29] += R_0_0_0_2 * dm_kl[7];
            vj_ij[2] += R_0_0_1_1 * dm_kl[0];
            vj_ij[6] += R_0_0_1_1 * dm_kl[1];
            vj_ij[10] += R_0_0_1_1 * dm_kl[2];
            vj_ij[14] += R_0_0_1_1 * dm_kl[3];
            vj_ij[18] += R_0_0_1_1 * dm_kl[4];
            vj_ij[22] += R_0_0_1_1 * dm_kl[5];
            vj_ij[26] += R_0_0_1_1 * dm_kl[6];
            vj_ij[30] += R_0_0_1_1 * dm_kl[7];
            vj_ij[3] += R_0_1_0_1 * dm_kl[0];
            vj_ij[7] += R_0_1_0_1 * dm_kl[1];
            vj_ij[11] += R_0_1_0_1 * dm_kl[2];
            vj_ij[15] += R_0_1_0_1 * dm_kl[3];
            vj_ij[19] += R_0_1_0_1 * dm_kl[4];
            vj_ij[23] += R_0_1_0_1 * dm_kl[5];
            vj_ij[27] += R_0_1_0_1 * dm_kl[6];
            vj_ij[31] += R_0_1_0_1 * dm_kl[7];
            for (int m = 0; m < 8; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2];
            }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[4] += R_0_0_1_0 * dm_kl[1];
            vj_ij[8] += R_0_0_1_0 * dm_kl[2];
            vj_ij[12] += R_0_0_1_0 * dm_kl[3];
            vj_ij[16] += R_0_0_1_0 * dm_kl[4];
            vj_ij[20] += R_0_0_1_0 * dm_kl[5];
            vj_ij[24] += R_0_0_1_0 * dm_kl[6];
            vj_ij[28] += R_0_0_1_0 * dm_kl[7];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[5] += R_0_0_1_1 * dm_kl[1];
            vj_ij[9] += R_0_0_1_1 * dm_kl[2];
            vj_ij[13] += R_0_0_1_1 * dm_kl[3];
            vj_ij[17] += R_0_0_1_1 * dm_kl[4];
            vj_ij[21] += R_0_0_1_1 * dm_kl[5];
            vj_ij[25] += R_0_0_1_1 * dm_kl[6];
            vj_ij[29] += R_0_0_1_1 * dm_kl[7];
            vj_ij[2] += R_0_0_2_0 * dm_kl[0];
            vj_ij[6] += R_0_0_2_0 * dm_kl[1];
            vj_ij[10] += R_0_0_2_0 * dm_kl[2];
            vj_ij[14] += R_0_0_2_0 * dm_kl[3];
            vj_ij[18] += R_0_0_2_0 * dm_kl[4];
            vj_ij[22] += R_0_0_2_0 * dm_kl[5];
            vj_ij[26] += R_0_0_2_0 * dm_kl[6];
            vj_ij[30] += R_0_0_2_0 * dm_kl[7];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[7] += R_0_1_1_0 * dm_kl[1];
            vj_ij[11] += R_0_1_1_0 * dm_kl[2];
            vj_ij[15] += R_0_1_1_0 * dm_kl[3];
            vj_ij[19] += R_0_1_1_0 * dm_kl[4];
            vj_ij[23] += R_0_1_1_0 * dm_kl[5];
            vj_ij[27] += R_0_1_1_0 * dm_kl[6];
            vj_ij[31] += R_0_1_1_0 * dm_kl[7];
            for (int m = 0; m < 8; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3];
            }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[4] += R_0_1_0_0 * dm_kl[1];
            vj_ij[8] += R_0_1_0_0 * dm_kl[2];
            vj_ij[12] += R_0_1_0_0 * dm_kl[3];
            vj_ij[16] += R_0_1_0_0 * dm_kl[4];
            vj_ij[20] += R_0_1_0_0 * dm_kl[5];
            vj_ij[24] += R_0_1_0_0 * dm_kl[6];
            vj_ij[28] += R_0_1_0_0 * dm_kl[7];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[5] += R_0_1_0_1 * dm_kl[1];
            vj_ij[9] += R_0_1_0_1 * dm_kl[2];
            vj_ij[13] += R_0_1_0_1 * dm_kl[3];
            vj_ij[17] += R_0_1_0_1 * dm_kl[4];
            vj_ij[21] += R_0_1_0_1 * dm_kl[5];
            vj_ij[25] += R_0_1_0_1 * dm_kl[6];
            vj_ij[29] += R_0_1_0_1 * dm_kl[7];
            vj_ij[2] += R_0_1_1_0 * dm_kl[0];
            vj_ij[6] += R_0_1_1_0 * dm_kl[1];
            vj_ij[10] += R_0_1_1_0 * dm_kl[2];
            vj_ij[14] += R_0_1_1_0 * dm_kl[3];
            vj_ij[18] += R_0_1_1_0 * dm_kl[4];
            vj_ij[22] += R_0_1_1_0 * dm_kl[5];
            vj_ij[26] += R_0_1_1_0 * dm_kl[6];
            vj_ij[30] += R_0_1_1_0 * dm_kl[7];
            vj_ij[3] += R_0_2_0_0 * dm_kl[0];
            vj_ij[7] += R_0_2_0_0 * dm_kl[1];
            vj_ij[11] += R_0_2_0_0 * dm_kl[2];
            vj_ij[15] += R_0_2_0_0 * dm_kl[3];
            vj_ij[19] += R_0_2_0_0 * dm_kl[4];
            vj_ij[23] += R_0_2_0_0 * dm_kl[5];
            vj_ij[27] += R_0_2_0_0 * dm_kl[6];
            vj_ij[31] += R_0_2_0_0 * dm_kl[7];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+4];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                __syncthreads();
                for (int m = 0; m < 8; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+4*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 8; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 24; n += 16) {
        int kl = n / 6;
        int batch_kl = n - kl * 6;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 96 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*384+kl*96]);
            }
        }
    }
} }

// TILEX=48, TILEY=16
__global__ static
void md_j_4dm_2_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
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
    double vj_kl[8];
    double dm_kl[8];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2048;
    double *Rp_cache = Rq_cache + 1024;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 1344 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 1088; n += 256) {
        Rq_cache[n] = 0.;
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
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+256] = 1e5;
            Rq_cache[n+512] = 1e5;
            Rq_cache[n+768] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 8) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 2048; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 10 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 10;
            int i = n - i_dm * 10;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[80];
        for (int ij = 0; ij < 80; ++ij) {
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
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*256+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[160];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[176];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[192];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[208];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[224];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[240];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[256];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[272];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[288];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*256+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[11] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[12] += R_0_0_0_2 * dm_kl[1];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[13] += R_0_0_1_0 * dm_kl[1];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[14] += R_0_0_1_1 * dm_kl[1];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[15] += R_0_0_2_0 * dm_kl[1];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[16] += R_0_1_0_0 * dm_kl[1];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[17] += R_0_1_0_1 * dm_kl[1];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[18] += R_0_1_1_0 * dm_kl[1];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            vj_ij[19] += R_0_2_0_0 * dm_kl[1];
            } else {
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[160];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[320];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[480];
            vj_kl[4] += gamma_inc[0*256] * dm_ij_cache[640];
            vj_kl[5] += gamma_inc[0*256] * dm_ij_cache[800];
            vj_kl[6] += gamma_inc[0*256] * dm_ij_cache[960];
            vj_kl[7] += gamma_inc[0*256] * dm_ij_cache[1120];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[176];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[336];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[496];
            vj_kl[4] += R_0_0_0_1 * dm_ij_cache[656];
            vj_kl[5] += R_0_0_0_1 * dm_ij_cache[816];
            vj_kl[6] += R_0_0_0_1 * dm_ij_cache[976];
            vj_kl[7] += R_0_0_0_1 * dm_ij_cache[1136];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[192];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[352];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[512];
            vj_kl[4] += R_0_0_0_2 * dm_ij_cache[672];
            vj_kl[5] += R_0_0_0_2 * dm_ij_cache[832];
            vj_kl[6] += R_0_0_0_2 * dm_ij_cache[992];
            vj_kl[7] += R_0_0_0_2 * dm_ij_cache[1152];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[208];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[368];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[528];
            vj_kl[4] += R_0_0_1_0 * dm_ij_cache[688];
            vj_kl[5] += R_0_0_1_0 * dm_ij_cache[848];
            vj_kl[6] += R_0_0_1_0 * dm_ij_cache[1008];
            vj_kl[7] += R_0_0_1_0 * dm_ij_cache[1168];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[224];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[384];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[544];
            vj_kl[4] += R_0_0_1_1 * dm_ij_cache[704];
            vj_kl[5] += R_0_0_1_1 * dm_ij_cache[864];
            vj_kl[6] += R_0_0_1_1 * dm_ij_cache[1024];
            vj_kl[7] += R_0_0_1_1 * dm_ij_cache[1184];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[240];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[400];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[560];
            vj_kl[4] += R_0_0_2_0 * dm_ij_cache[720];
            vj_kl[5] += R_0_0_2_0 * dm_ij_cache[880];
            vj_kl[6] += R_0_0_2_0 * dm_ij_cache[1040];
            vj_kl[7] += R_0_0_2_0 * dm_ij_cache[1200];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[256];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[416];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[576];
            vj_kl[4] += R_0_1_0_0 * dm_ij_cache[736];
            vj_kl[5] += R_0_1_0_0 * dm_ij_cache[896];
            vj_kl[6] += R_0_1_0_0 * dm_ij_cache[1056];
            vj_kl[7] += R_0_1_0_0 * dm_ij_cache[1216];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[272];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[432];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[592];
            vj_kl[4] += R_0_1_0_1 * dm_ij_cache[752];
            vj_kl[5] += R_0_1_0_1 * dm_ij_cache[912];
            vj_kl[6] += R_0_1_0_1 * dm_ij_cache[1072];
            vj_kl[7] += R_0_1_0_1 * dm_ij_cache[1232];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[288];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[448];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[608];
            vj_kl[4] += R_0_1_1_0 * dm_ij_cache[768];
            vj_kl[5] += R_0_1_1_0 * dm_ij_cache[928];
            vj_kl[6] += R_0_1_1_0 * dm_ij_cache[1088];
            vj_kl[7] += R_0_1_1_0 * dm_ij_cache[1248];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[304];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[464];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[624];
            vj_kl[4] += R_0_2_0_0 * dm_ij_cache[784];
            vj_kl[5] += R_0_2_0_0 * dm_ij_cache[944];
            vj_kl[6] += R_0_2_0_0 * dm_ij_cache[1104];
            vj_kl[7] += R_0_2_0_0 * dm_ij_cache[1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*256+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[30] += gamma_inc[0*256] * dm_kl[3];
            vj_ij[40] += gamma_inc[0*256] * dm_kl[4];
            vj_ij[50] += gamma_inc[0*256] * dm_kl[5];
            vj_ij[60] += gamma_inc[0*256] * dm_kl[6];
            vj_ij[70] += gamma_inc[0*256] * dm_kl[7];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[11] += R_0_0_0_1 * dm_kl[1];
            vj_ij[21] += R_0_0_0_1 * dm_kl[2];
            vj_ij[31] += R_0_0_0_1 * dm_kl[3];
            vj_ij[41] += R_0_0_0_1 * dm_kl[4];
            vj_ij[51] += R_0_0_0_1 * dm_kl[5];
            vj_ij[61] += R_0_0_0_1 * dm_kl[6];
            vj_ij[71] += R_0_0_0_1 * dm_kl[7];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[12] += R_0_0_0_2 * dm_kl[1];
            vj_ij[22] += R_0_0_0_2 * dm_kl[2];
            vj_ij[32] += R_0_0_0_2 * dm_kl[3];
            vj_ij[42] += R_0_0_0_2 * dm_kl[4];
            vj_ij[52] += R_0_0_0_2 * dm_kl[5];
            vj_ij[62] += R_0_0_0_2 * dm_kl[6];
            vj_ij[72] += R_0_0_0_2 * dm_kl[7];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[13] += R_0_0_1_0 * dm_kl[1];
            vj_ij[23] += R_0_0_1_0 * dm_kl[2];
            vj_ij[33] += R_0_0_1_0 * dm_kl[3];
            vj_ij[43] += R_0_0_1_0 * dm_kl[4];
            vj_ij[53] += R_0_0_1_0 * dm_kl[5];
            vj_ij[63] += R_0_0_1_0 * dm_kl[6];
            vj_ij[73] += R_0_0_1_0 * dm_kl[7];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[14] += R_0_0_1_1 * dm_kl[1];
            vj_ij[24] += R_0_0_1_1 * dm_kl[2];
            vj_ij[34] += R_0_0_1_1 * dm_kl[3];
            vj_ij[44] += R_0_0_1_1 * dm_kl[4];
            vj_ij[54] += R_0_0_1_1 * dm_kl[5];
            vj_ij[64] += R_0_0_1_1 * dm_kl[6];
            vj_ij[74] += R_0_0_1_1 * dm_kl[7];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[15] += R_0_0_2_0 * dm_kl[1];
            vj_ij[25] += R_0_0_2_0 * dm_kl[2];
            vj_ij[35] += R_0_0_2_0 * dm_kl[3];
            vj_ij[45] += R_0_0_2_0 * dm_kl[4];
            vj_ij[55] += R_0_0_2_0 * dm_kl[5];
            vj_ij[65] += R_0_0_2_0 * dm_kl[6];
            vj_ij[75] += R_0_0_2_0 * dm_kl[7];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[16] += R_0_1_0_0 * dm_kl[1];
            vj_ij[26] += R_0_1_0_0 * dm_kl[2];
            vj_ij[36] += R_0_1_0_0 * dm_kl[3];
            vj_ij[46] += R_0_1_0_0 * dm_kl[4];
            vj_ij[56] += R_0_1_0_0 * dm_kl[5];
            vj_ij[66] += R_0_1_0_0 * dm_kl[6];
            vj_ij[76] += R_0_1_0_0 * dm_kl[7];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[17] += R_0_1_0_1 * dm_kl[1];
            vj_ij[27] += R_0_1_0_1 * dm_kl[2];
            vj_ij[37] += R_0_1_0_1 * dm_kl[3];
            vj_ij[47] += R_0_1_0_1 * dm_kl[4];
            vj_ij[57] += R_0_1_0_1 * dm_kl[5];
            vj_ij[67] += R_0_1_0_1 * dm_kl[6];
            vj_ij[77] += R_0_1_0_1 * dm_kl[7];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[18] += R_0_1_1_0 * dm_kl[1];
            vj_ij[28] += R_0_1_1_0 * dm_kl[2];
            vj_ij[38] += R_0_1_1_0 * dm_kl[3];
            vj_ij[48] += R_0_1_1_0 * dm_kl[4];
            vj_ij[58] += R_0_1_1_0 * dm_kl[5];
            vj_ij[68] += R_0_1_1_0 * dm_kl[6];
            vj_ij[78] += R_0_1_1_0 * dm_kl[7];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            vj_ij[19] += R_0_2_0_0 * dm_kl[1];
            vj_ij[29] += R_0_2_0_0 * dm_kl[2];
            vj_ij[39] += R_0_2_0_0 * dm_kl[3];
            vj_ij[49] += R_0_2_0_0 * dm_kl[4];
            vj_ij[59] += R_0_2_0_0 * dm_kl[5];
            vj_ij[69] += R_0_2_0_0 * dm_kl[6];
            vj_ij[79] += R_0_2_0_0 * dm_kl[7];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 10; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+10];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 10; ++n) {
                __syncthreads();
                for (int m = 0; m < 8; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+10*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 8; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
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
            for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*256+kl*256]);
            }
        }
    }
} }

// TILEX=48, TILEY=10
__global__ static
void md_j_4dm_2_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 160;
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
    double vj_kl[4];
    double dm_kl[4];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2560;
    double *Rp_cache = Rq_cache + 640;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 704 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 704; n += 256) {
        Rq_cache[n] = 0.;
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
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+160] = 1e5;
            Rq_cache[n+320] = 1e5;
            Rq_cache[n+480] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 2560; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 10 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 10;
            int i = n - i_dm * 10;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[40];
        for (int ij = 0; ij < 40; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 10; ++batch_kl) {
            int task_kl0 = blockIdx.y * 160 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*10] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            boys_fn(gamma_inc, theta, rr, omega, fac, 3, 0, 256);
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+160] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[64];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+320] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[128];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+480] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+1];
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[3] += R_0_0_1_1 * dm_kl[0];
            vj_ij[4] += R_0_0_1_2 * dm_kl[0];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[6] += R_0_1_0_1 * dm_kl[0];
            vj_ij[7] += R_0_1_0_2 * dm_kl[0];
            vj_ij[8] += R_0_1_1_1 * dm_kl[0];
            vj_ij[9] += R_0_2_0_1 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+2];
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[3] += R_0_0_2_0 * dm_kl[0];
            vj_ij[4] += R_0_0_2_1 * dm_kl[0];
            vj_ij[5] += R_0_0_3_0 * dm_kl[0];
            vj_ij[6] += R_0_1_1_0 * dm_kl[0];
            vj_ij[7] += R_0_1_1_1 * dm_kl[0];
            vj_ij[8] += R_0_1_2_0 * dm_kl[0];
            vj_ij[9] += R_0_2_1_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+3];
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[4] += R_0_1_1_1 * dm_kl[0];
            vj_ij[5] += R_0_1_2_0 * dm_kl[0];
            vj_ij[6] += R_0_2_0_0 * dm_kl[0];
            vj_ij[7] += R_0_2_0_1 * dm_kl[0];
            vj_ij[8] += R_0_2_1_0 * dm_kl[0];
            vj_ij[9] += R_0_3_0_0 * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[160];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[176];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[192];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[208];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[224];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[240];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[256];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[272];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[288];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[160];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[176];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[192];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[208];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[224];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[240];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[272];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[288];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+160] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[160];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[176];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[192];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[208];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[224];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[256];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[272];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[288];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+320] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[288];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+480] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[11] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[12] += R_0_0_0_2 * dm_kl[1];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[13] += R_0_0_1_0 * dm_kl[1];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[14] += R_0_0_1_1 * dm_kl[1];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[15] += R_0_0_2_0 * dm_kl[1];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[16] += R_0_1_0_0 * dm_kl[1];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[17] += R_0_1_0_1 * dm_kl[1];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[18] += R_0_1_1_0 * dm_kl[1];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            vj_ij[19] += R_0_2_0_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[10] += R_0_0_0_1 * dm_kl[1];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[11] += R_0_0_0_2 * dm_kl[1];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[12] += R_0_0_0_3 * dm_kl[1];
            vj_ij[3] += R_0_0_1_1 * dm_kl[0];
            vj_ij[13] += R_0_0_1_1 * dm_kl[1];
            vj_ij[4] += R_0_0_1_2 * dm_kl[0];
            vj_ij[14] += R_0_0_1_2 * dm_kl[1];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[15] += R_0_0_2_1 * dm_kl[1];
            vj_ij[6] += R_0_1_0_1 * dm_kl[0];
            vj_ij[16] += R_0_1_0_1 * dm_kl[1];
            vj_ij[7] += R_0_1_0_2 * dm_kl[0];
            vj_ij[17] += R_0_1_0_2 * dm_kl[1];
            vj_ij[8] += R_0_1_1_1 * dm_kl[0];
            vj_ij[18] += R_0_1_1_1 * dm_kl[1];
            vj_ij[9] += R_0_2_0_1 * dm_kl[0];
            vj_ij[19] += R_0_2_0_1 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2]; }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[10] += R_0_0_1_0 * dm_kl[1];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[11] += R_0_0_1_1 * dm_kl[1];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[12] += R_0_0_1_2 * dm_kl[1];
            vj_ij[3] += R_0_0_2_0 * dm_kl[0];
            vj_ij[13] += R_0_0_2_0 * dm_kl[1];
            vj_ij[4] += R_0_0_2_1 * dm_kl[0];
            vj_ij[14] += R_0_0_2_1 * dm_kl[1];
            vj_ij[5] += R_0_0_3_0 * dm_kl[0];
            vj_ij[15] += R_0_0_3_0 * dm_kl[1];
            vj_ij[6] += R_0_1_1_0 * dm_kl[0];
            vj_ij[16] += R_0_1_1_0 * dm_kl[1];
            vj_ij[7] += R_0_1_1_1 * dm_kl[0];
            vj_ij[17] += R_0_1_1_1 * dm_kl[1];
            vj_ij[8] += R_0_1_2_0 * dm_kl[0];
            vj_ij[18] += R_0_1_2_0 * dm_kl[1];
            vj_ij[9] += R_0_2_1_0 * dm_kl[0];
            vj_ij[19] += R_0_2_1_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[10] += R_0_1_0_0 * dm_kl[1];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[11] += R_0_1_0_1 * dm_kl[1];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[12] += R_0_1_0_2 * dm_kl[1];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[13] += R_0_1_1_0 * dm_kl[1];
            vj_ij[4] += R_0_1_1_1 * dm_kl[0];
            vj_ij[14] += R_0_1_1_1 * dm_kl[1];
            vj_ij[5] += R_0_1_2_0 * dm_kl[0];
            vj_ij[15] += R_0_1_2_0 * dm_kl[1];
            vj_ij[6] += R_0_2_0_0 * dm_kl[0];
            vj_ij[16] += R_0_2_0_0 * dm_kl[1];
            vj_ij[7] += R_0_2_0_1 * dm_kl[0];
            vj_ij[17] += R_0_2_0_1 * dm_kl[1];
            vj_ij[8] += R_0_2_1_0 * dm_kl[0];
            vj_ij[18] += R_0_2_1_0 * dm_kl[1];
            vj_ij[9] += R_0_3_0_0 * dm_kl[0];
            vj_ij[19] += R_0_3_0_0 * dm_kl[1];
            } else {
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[160];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[320];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[480];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[176];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[336];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[496];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[192];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[352];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[512];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[208];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[368];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[528];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[224];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[384];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[544];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[240];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[400];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[560];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[256];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[416];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[576];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[272];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[432];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[592];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[288];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[448];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[608];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[304];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[464];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[160];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[320];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[480];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[176];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[336];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[496];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[192];
            vj_kl[2] -= R_0_0_0_3 * dm_ij_cache[352];
            vj_kl[3] -= R_0_0_0_3 * dm_ij_cache[512];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[208];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[368];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[528];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[224];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[384];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[544];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[240];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[400];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[560];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[256];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[416];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[576];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[272];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[432];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[592];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[288];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[448];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[608];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[304];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[464];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+160] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[160];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[320];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[176];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[336];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[496];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[192];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[352];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[512];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[208];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[368];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[528];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[224];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[384];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[544];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[240];
            vj_kl[2] -= R_0_0_3_0 * dm_ij_cache[400];
            vj_kl[3] -= R_0_0_3_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[256];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[416];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[576];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[272];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[432];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[592];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[288];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[448];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[608];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[304];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[464];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+320] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[320];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[336];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[496];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[352];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[512];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[368];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[528];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[384];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[544];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[400];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[416];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[576];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[432];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[592];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[288];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[448];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[608];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[304];
            vj_kl[2] -= R_0_3_0_0 * dm_ij_cache[464];
            vj_kl[3] -= R_0_3_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+480] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[30] += gamma_inc[0*256] * dm_kl[3];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[11] += R_0_0_0_1 * dm_kl[1];
            vj_ij[21] += R_0_0_0_1 * dm_kl[2];
            vj_ij[31] += R_0_0_0_1 * dm_kl[3];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[12] += R_0_0_0_2 * dm_kl[1];
            vj_ij[22] += R_0_0_0_2 * dm_kl[2];
            vj_ij[32] += R_0_0_0_2 * dm_kl[3];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[13] += R_0_0_1_0 * dm_kl[1];
            vj_ij[23] += R_0_0_1_0 * dm_kl[2];
            vj_ij[33] += R_0_0_1_0 * dm_kl[3];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[14] += R_0_0_1_1 * dm_kl[1];
            vj_ij[24] += R_0_0_1_1 * dm_kl[2];
            vj_ij[34] += R_0_0_1_1 * dm_kl[3];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[15] += R_0_0_2_0 * dm_kl[1];
            vj_ij[25] += R_0_0_2_0 * dm_kl[2];
            vj_ij[35] += R_0_0_2_0 * dm_kl[3];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[16] += R_0_1_0_0 * dm_kl[1];
            vj_ij[26] += R_0_1_0_0 * dm_kl[2];
            vj_ij[36] += R_0_1_0_0 * dm_kl[3];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[17] += R_0_1_0_1 * dm_kl[1];
            vj_ij[27] += R_0_1_0_1 * dm_kl[2];
            vj_ij[37] += R_0_1_0_1 * dm_kl[3];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[18] += R_0_1_1_0 * dm_kl[1];
            vj_ij[28] += R_0_1_1_0 * dm_kl[2];
            vj_ij[38] += R_0_1_1_0 * dm_kl[3];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            vj_ij[19] += R_0_2_0_0 * dm_kl[1];
            vj_ij[29] += R_0_2_0_0 * dm_kl[2];
            vj_ij[39] += R_0_2_0_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1];
            }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[10] += R_0_0_0_1 * dm_kl[1];
            vj_ij[20] += R_0_0_0_1 * dm_kl[2];
            vj_ij[30] += R_0_0_0_1 * dm_kl[3];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[11] += R_0_0_0_2 * dm_kl[1];
            vj_ij[21] += R_0_0_0_2 * dm_kl[2];
            vj_ij[31] += R_0_0_0_2 * dm_kl[3];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[12] += R_0_0_0_3 * dm_kl[1];
            vj_ij[22] += R_0_0_0_3 * dm_kl[2];
            vj_ij[32] += R_0_0_0_3 * dm_kl[3];
            vj_ij[3] += R_0_0_1_1 * dm_kl[0];
            vj_ij[13] += R_0_0_1_1 * dm_kl[1];
            vj_ij[23] += R_0_0_1_1 * dm_kl[2];
            vj_ij[33] += R_0_0_1_1 * dm_kl[3];
            vj_ij[4] += R_0_0_1_2 * dm_kl[0];
            vj_ij[14] += R_0_0_1_2 * dm_kl[1];
            vj_ij[24] += R_0_0_1_2 * dm_kl[2];
            vj_ij[34] += R_0_0_1_2 * dm_kl[3];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[15] += R_0_0_2_1 * dm_kl[1];
            vj_ij[25] += R_0_0_2_1 * dm_kl[2];
            vj_ij[35] += R_0_0_2_1 * dm_kl[3];
            vj_ij[6] += R_0_1_0_1 * dm_kl[0];
            vj_ij[16] += R_0_1_0_1 * dm_kl[1];
            vj_ij[26] += R_0_1_0_1 * dm_kl[2];
            vj_ij[36] += R_0_1_0_1 * dm_kl[3];
            vj_ij[7] += R_0_1_0_2 * dm_kl[0];
            vj_ij[17] += R_0_1_0_2 * dm_kl[1];
            vj_ij[27] += R_0_1_0_2 * dm_kl[2];
            vj_ij[37] += R_0_1_0_2 * dm_kl[3];
            vj_ij[8] += R_0_1_1_1 * dm_kl[0];
            vj_ij[18] += R_0_1_1_1 * dm_kl[1];
            vj_ij[28] += R_0_1_1_1 * dm_kl[2];
            vj_ij[38] += R_0_1_1_1 * dm_kl[3];
            vj_ij[9] += R_0_2_0_1 * dm_kl[0];
            vj_ij[19] += R_0_2_0_1 * dm_kl[1];
            vj_ij[29] += R_0_2_0_1 * dm_kl[2];
            vj_ij[39] += R_0_2_0_1 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2];
            }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[10] += R_0_0_1_0 * dm_kl[1];
            vj_ij[20] += R_0_0_1_0 * dm_kl[2];
            vj_ij[30] += R_0_0_1_0 * dm_kl[3];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[11] += R_0_0_1_1 * dm_kl[1];
            vj_ij[21] += R_0_0_1_1 * dm_kl[2];
            vj_ij[31] += R_0_0_1_1 * dm_kl[3];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[12] += R_0_0_1_2 * dm_kl[1];
            vj_ij[22] += R_0_0_1_2 * dm_kl[2];
            vj_ij[32] += R_0_0_1_2 * dm_kl[3];
            vj_ij[3] += R_0_0_2_0 * dm_kl[0];
            vj_ij[13] += R_0_0_2_0 * dm_kl[1];
            vj_ij[23] += R_0_0_2_0 * dm_kl[2];
            vj_ij[33] += R_0_0_2_0 * dm_kl[3];
            vj_ij[4] += R_0_0_2_1 * dm_kl[0];
            vj_ij[14] += R_0_0_2_1 * dm_kl[1];
            vj_ij[24] += R_0_0_2_1 * dm_kl[2];
            vj_ij[34] += R_0_0_2_1 * dm_kl[3];
            vj_ij[5] += R_0_0_3_0 * dm_kl[0];
            vj_ij[15] += R_0_0_3_0 * dm_kl[1];
            vj_ij[25] += R_0_0_3_0 * dm_kl[2];
            vj_ij[35] += R_0_0_3_0 * dm_kl[3];
            vj_ij[6] += R_0_1_1_0 * dm_kl[0];
            vj_ij[16] += R_0_1_1_0 * dm_kl[1];
            vj_ij[26] += R_0_1_1_0 * dm_kl[2];
            vj_ij[36] += R_0_1_1_0 * dm_kl[3];
            vj_ij[7] += R_0_1_1_1 * dm_kl[0];
            vj_ij[17] += R_0_1_1_1 * dm_kl[1];
            vj_ij[27] += R_0_1_1_1 * dm_kl[2];
            vj_ij[37] += R_0_1_1_1 * dm_kl[3];
            vj_ij[8] += R_0_1_2_0 * dm_kl[0];
            vj_ij[18] += R_0_1_2_0 * dm_kl[1];
            vj_ij[28] += R_0_1_2_0 * dm_kl[2];
            vj_ij[38] += R_0_1_2_0 * dm_kl[3];
            vj_ij[9] += R_0_2_1_0 * dm_kl[0];
            vj_ij[19] += R_0_2_1_0 * dm_kl[1];
            vj_ij[29] += R_0_2_1_0 * dm_kl[2];
            vj_ij[39] += R_0_2_1_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3];
            }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[10] += R_0_1_0_0 * dm_kl[1];
            vj_ij[20] += R_0_1_0_0 * dm_kl[2];
            vj_ij[30] += R_0_1_0_0 * dm_kl[3];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[11] += R_0_1_0_1 * dm_kl[1];
            vj_ij[21] += R_0_1_0_1 * dm_kl[2];
            vj_ij[31] += R_0_1_0_1 * dm_kl[3];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[12] += R_0_1_0_2 * dm_kl[1];
            vj_ij[22] += R_0_1_0_2 * dm_kl[2];
            vj_ij[32] += R_0_1_0_2 * dm_kl[3];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[13] += R_0_1_1_0 * dm_kl[1];
            vj_ij[23] += R_0_1_1_0 * dm_kl[2];
            vj_ij[33] += R_0_1_1_0 * dm_kl[3];
            vj_ij[4] += R_0_1_1_1 * dm_kl[0];
            vj_ij[14] += R_0_1_1_1 * dm_kl[1];
            vj_ij[24] += R_0_1_1_1 * dm_kl[2];
            vj_ij[34] += R_0_1_1_1 * dm_kl[3];
            vj_ij[5] += R_0_1_2_0 * dm_kl[0];
            vj_ij[15] += R_0_1_2_0 * dm_kl[1];
            vj_ij[25] += R_0_1_2_0 * dm_kl[2];
            vj_ij[35] += R_0_1_2_0 * dm_kl[3];
            vj_ij[6] += R_0_2_0_0 * dm_kl[0];
            vj_ij[16] += R_0_2_0_0 * dm_kl[1];
            vj_ij[26] += R_0_2_0_0 * dm_kl[2];
            vj_ij[36] += R_0_2_0_0 * dm_kl[3];
            vj_ij[7] += R_0_2_0_1 * dm_kl[0];
            vj_ij[17] += R_0_2_0_1 * dm_kl[1];
            vj_ij[27] += R_0_2_0_1 * dm_kl[2];
            vj_ij[37] += R_0_2_0_1 * dm_kl[3];
            vj_ij[8] += R_0_2_1_0 * dm_kl[0];
            vj_ij[18] += R_0_2_1_0 * dm_kl[1];
            vj_ij[28] += R_0_2_1_0 * dm_kl[2];
            vj_ij[38] += R_0_2_1_0 * dm_kl[3];
            vj_ij[9] += R_0_3_0_0 * dm_kl[0];
            vj_ij[19] += R_0_3_0_0 * dm_kl[1];
            vj_ij[29] += R_0_3_0_0 * dm_kl[2];
            vj_ij[39] += R_0_3_0_0 * dm_kl[3];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 10; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+10];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 10; ++n) {
                __syncthreads();
                for (int m = 0; m < 4; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+10*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 4; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
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
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*640+kl*160]);
            }
        }
    }
} }

// TILEX=4, TILEY=4
__global__ static
void md_j_4dm_2_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 64;
    int task_kl0 = blockIdx.y * 64;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+64 <= task_kl0) {
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
    double vj_kl[4];
    double dm_kl[4];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2560;
    double *Rp_cache = Rq_cache + 256;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 704 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 320; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 64; n += 256) {
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
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+64] = 1e5;
            Rq_cache[n+128] = 1e5;
            Rq_cache[n+192] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 2560; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    for (int batch_ij = 0; batch_ij < 4; ++batch_ij) {
        int task_ij0 = blockIdx.x * 64 + batch_ij * 16;
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
        int nf3ij_dm = 10 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 10;
            int i = n - i_dm * 10;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[40];
        for (int ij = 0; ij < 40; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 4; ++batch_kl) {
            int task_kl0 = blockIdx.y * 64 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*4] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*4] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            double ykl = Rq_cache[sq_kl+64];
            double zkl = Rq_cache[sq_kl+128];
            double akl = Rq_cache[sq_kl+192];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, 0, 256);
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+64] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[0];
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[16];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[32];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[48];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[64];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[80];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[96];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[112];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[128];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+128] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[64];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+192] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[0];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[16];
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[32];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[48];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[64];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[80];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[96];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[112];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[128];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+256] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[0];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[16];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[32];
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[48];
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[64];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[80];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[96];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[112];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[128];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+320] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[128];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+384] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[0];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[16];
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[32];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[48];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[64];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[80];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[96];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[112];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[128];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+448] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[0];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[16];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[32];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[48];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[64];
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[80];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[96];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[112];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[128];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+512] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[0];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[16];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[32];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[48];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[64];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[80];
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[96];
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[112];
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[128];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+576] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+1];
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[3] += R_0_0_1_1 * dm_kl[0];
            vj_ij[4] += R_0_0_1_2 * dm_kl[0];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[6] += R_0_1_0_1 * dm_kl[0];
            vj_ij[7] += R_0_1_0_2 * dm_kl[0];
            vj_ij[8] += R_0_1_1_1 * dm_kl[0];
            vj_ij[9] += R_0_2_0_1 * dm_kl[0];
            dm_kl[0] = 1 * dm[kl_loc0+2];
            vj_ij[0] += R_0_0_0_2 * dm_kl[0];
            vj_ij[1] += R_0_0_0_3 * dm_kl[0];
            vj_ij[2] += R_0_0_0_4 * dm_kl[0];
            vj_ij[3] += R_0_0_1_2 * dm_kl[0];
            vj_ij[4] += R_0_0_1_3 * dm_kl[0];
            vj_ij[5] += R_0_0_2_2 * dm_kl[0];
            vj_ij[6] += R_0_1_0_2 * dm_kl[0];
            vj_ij[7] += R_0_1_0_3 * dm_kl[0];
            vj_ij[8] += R_0_1_1_2 * dm_kl[0];
            vj_ij[9] += R_0_2_0_2 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+3];
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[3] += R_0_0_2_0 * dm_kl[0];
            vj_ij[4] += R_0_0_2_1 * dm_kl[0];
            vj_ij[5] += R_0_0_3_0 * dm_kl[0];
            vj_ij[6] += R_0_1_1_0 * dm_kl[0];
            vj_ij[7] += R_0_1_1_1 * dm_kl[0];
            vj_ij[8] += R_0_1_2_0 * dm_kl[0];
            vj_ij[9] += R_0_2_1_0 * dm_kl[0];
            dm_kl[0] = 1 * dm[kl_loc0+4];
            vj_ij[0] += R_0_0_1_1 * dm_kl[0];
            vj_ij[1] += R_0_0_1_2 * dm_kl[0];
            vj_ij[2] += R_0_0_1_3 * dm_kl[0];
            vj_ij[3] += R_0_0_2_1 * dm_kl[0];
            vj_ij[4] += R_0_0_2_2 * dm_kl[0];
            vj_ij[5] += R_0_0_3_1 * dm_kl[0];
            vj_ij[6] += R_0_1_1_1 * dm_kl[0];
            vj_ij[7] += R_0_1_1_2 * dm_kl[0];
            vj_ij[8] += R_0_1_2_1 * dm_kl[0];
            vj_ij[9] += R_0_2_1_1 * dm_kl[0];
            dm_kl[0] = 1 * dm[kl_loc0+5];
            vj_ij[0] += R_0_0_2_0 * dm_kl[0];
            vj_ij[1] += R_0_0_2_1 * dm_kl[0];
            vj_ij[2] += R_0_0_2_2 * dm_kl[0];
            vj_ij[3] += R_0_0_3_0 * dm_kl[0];
            vj_ij[4] += R_0_0_3_1 * dm_kl[0];
            vj_ij[5] += R_0_0_4_0 * dm_kl[0];
            vj_ij[6] += R_0_1_2_0 * dm_kl[0];
            vj_ij[7] += R_0_1_2_1 * dm_kl[0];
            vj_ij[8] += R_0_1_3_0 * dm_kl[0];
            vj_ij[9] += R_0_2_2_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+6];
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[4] += R_0_1_1_1 * dm_kl[0];
            vj_ij[5] += R_0_1_2_0 * dm_kl[0];
            vj_ij[6] += R_0_2_0_0 * dm_kl[0];
            vj_ij[7] += R_0_2_0_1 * dm_kl[0];
            vj_ij[8] += R_0_2_1_0 * dm_kl[0];
            vj_ij[9] += R_0_3_0_0 * dm_kl[0];
            dm_kl[0] = 1 * dm[kl_loc0+7];
            vj_ij[0] += R_0_1_0_1 * dm_kl[0];
            vj_ij[1] += R_0_1_0_2 * dm_kl[0];
            vj_ij[2] += R_0_1_0_3 * dm_kl[0];
            vj_ij[3] += R_0_1_1_1 * dm_kl[0];
            vj_ij[4] += R_0_1_1_2 * dm_kl[0];
            vj_ij[5] += R_0_1_2_1 * dm_kl[0];
            vj_ij[6] += R_0_2_0_1 * dm_kl[0];
            vj_ij[7] += R_0_2_0_2 * dm_kl[0];
            vj_ij[8] += R_0_2_1_1 * dm_kl[0];
            vj_ij[9] += R_0_3_0_1 * dm_kl[0];
            dm_kl[0] = 1 * dm[kl_loc0+8];
            vj_ij[0] += R_0_1_1_0 * dm_kl[0];
            vj_ij[1] += R_0_1_1_1 * dm_kl[0];
            vj_ij[2] += R_0_1_1_2 * dm_kl[0];
            vj_ij[3] += R_0_1_2_0 * dm_kl[0];
            vj_ij[4] += R_0_1_2_1 * dm_kl[0];
            vj_ij[5] += R_0_1_3_0 * dm_kl[0];
            vj_ij[6] += R_0_2_1_0 * dm_kl[0];
            vj_ij[7] += R_0_2_1_1 * dm_kl[0];
            vj_ij[8] += R_0_2_2_0 * dm_kl[0];
            vj_ij[9] += R_0_3_1_0 * dm_kl[0];
            dm_kl[0] = 1 * dm[kl_loc0+9];
            vj_ij[0] += R_0_2_0_0 * dm_kl[0];
            vj_ij[1] += R_0_2_0_1 * dm_kl[0];
            vj_ij[2] += R_0_2_0_2 * dm_kl[0];
            vj_ij[3] += R_0_2_1_0 * dm_kl[0];
            vj_ij[4] += R_0_2_1_1 * dm_kl[0];
            vj_ij[5] += R_0_2_2_0 * dm_kl[0];
            vj_ij[6] += R_0_3_0_0 * dm_kl[0];
            vj_ij[7] += R_0_3_0_1 * dm_kl[0];
            vj_ij[8] += R_0_3_1_0 * dm_kl[0];
            vj_ij[9] += R_0_4_0_0 * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[160];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[176];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[192];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[208];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[224];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[240];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[256];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[272];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[288];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[160];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[176];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[192];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[208];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[224];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[240];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[272];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[288];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+64] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[0];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[160];
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[176];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_4 * dm_ij_cache[192];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[208];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[224];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[240];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[256];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[272];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[288];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+128] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[160];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[176];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[192];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[208];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[224];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[256];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[272];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[288];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+192] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[0];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[160];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[176];
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[192];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[208];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[224];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[240];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[256];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[272];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[288];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+256] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[0];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[160];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[176];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[192];
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[208];
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[224];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_4_0 * dm_ij_cache[240];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[256];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[272];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[288];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+320] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[288];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+384] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[0];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[160];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[16];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[176];
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[32];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[192];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[208];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[64];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[224];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[240];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[96];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[256];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[112];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[272];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[144];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+448] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[0];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[160];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[176];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[192];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[208];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[224];
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[240];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[256];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[272];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+512] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[0];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[160];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[176];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[192];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[208];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[224];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[240];
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[256];
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[272];
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[288];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_4_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*640+576] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[11] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[12] += R_0_0_0_2 * dm_kl[1];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[13] += R_0_0_1_0 * dm_kl[1];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[14] += R_0_0_1_1 * dm_kl[1];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[15] += R_0_0_2_0 * dm_kl[1];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[16] += R_0_1_0_0 * dm_kl[1];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[17] += R_0_1_0_1 * dm_kl[1];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[18] += R_0_1_1_0 * dm_kl[1];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            vj_ij[19] += R_0_2_0_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[10] += R_0_0_0_1 * dm_kl[1];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[11] += R_0_0_0_2 * dm_kl[1];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[12] += R_0_0_0_3 * dm_kl[1];
            vj_ij[3] += R_0_0_1_1 * dm_kl[0];
            vj_ij[13] += R_0_0_1_1 * dm_kl[1];
            vj_ij[4] += R_0_0_1_2 * dm_kl[0];
            vj_ij[14] += R_0_0_1_2 * dm_kl[1];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[15] += R_0_0_2_1 * dm_kl[1];
            vj_ij[6] += R_0_1_0_1 * dm_kl[0];
            vj_ij[16] += R_0_1_0_1 * dm_kl[1];
            vj_ij[7] += R_0_1_0_2 * dm_kl[0];
            vj_ij[17] += R_0_1_0_2 * dm_kl[1];
            vj_ij[8] += R_0_1_1_1 * dm_kl[0];
            vj_ij[18] += R_0_1_1_1 * dm_kl[1];
            vj_ij[9] += R_0_2_0_1 * dm_kl[0];
            vj_ij[19] += R_0_2_0_1 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+2]; }
            vj_ij[0] += R_0_0_0_2 * dm_kl[0];
            vj_ij[10] += R_0_0_0_2 * dm_kl[1];
            vj_ij[1] += R_0_0_0_3 * dm_kl[0];
            vj_ij[11] += R_0_0_0_3 * dm_kl[1];
            vj_ij[2] += R_0_0_0_4 * dm_kl[0];
            vj_ij[12] += R_0_0_0_4 * dm_kl[1];
            vj_ij[3] += R_0_0_1_2 * dm_kl[0];
            vj_ij[13] += R_0_0_1_2 * dm_kl[1];
            vj_ij[4] += R_0_0_1_3 * dm_kl[0];
            vj_ij[14] += R_0_0_1_3 * dm_kl[1];
            vj_ij[5] += R_0_0_2_2 * dm_kl[0];
            vj_ij[15] += R_0_0_2_2 * dm_kl[1];
            vj_ij[6] += R_0_1_0_2 * dm_kl[0];
            vj_ij[16] += R_0_1_0_2 * dm_kl[1];
            vj_ij[7] += R_0_1_0_3 * dm_kl[0];
            vj_ij[17] += R_0_1_0_3 * dm_kl[1];
            vj_ij[8] += R_0_1_1_2 * dm_kl[0];
            vj_ij[18] += R_0_1_1_2 * dm_kl[1];
            vj_ij[9] += R_0_2_0_2 * dm_kl[0];
            vj_ij[19] += R_0_2_0_2 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[10] += R_0_0_1_0 * dm_kl[1];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[11] += R_0_0_1_1 * dm_kl[1];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[12] += R_0_0_1_2 * dm_kl[1];
            vj_ij[3] += R_0_0_2_0 * dm_kl[0];
            vj_ij[13] += R_0_0_2_0 * dm_kl[1];
            vj_ij[4] += R_0_0_2_1 * dm_kl[0];
            vj_ij[14] += R_0_0_2_1 * dm_kl[1];
            vj_ij[5] += R_0_0_3_0 * dm_kl[0];
            vj_ij[15] += R_0_0_3_0 * dm_kl[1];
            vj_ij[6] += R_0_1_1_0 * dm_kl[0];
            vj_ij[16] += R_0_1_1_0 * dm_kl[1];
            vj_ij[7] += R_0_1_1_1 * dm_kl[0];
            vj_ij[17] += R_0_1_1_1 * dm_kl[1];
            vj_ij[8] += R_0_1_2_0 * dm_kl[0];
            vj_ij[18] += R_0_1_2_0 * dm_kl[1];
            vj_ij[9] += R_0_2_1_0 * dm_kl[0];
            vj_ij[19] += R_0_2_1_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+4]; }
            vj_ij[0] += R_0_0_1_1 * dm_kl[0];
            vj_ij[10] += R_0_0_1_1 * dm_kl[1];
            vj_ij[1] += R_0_0_1_2 * dm_kl[0];
            vj_ij[11] += R_0_0_1_2 * dm_kl[1];
            vj_ij[2] += R_0_0_1_3 * dm_kl[0];
            vj_ij[12] += R_0_0_1_3 * dm_kl[1];
            vj_ij[3] += R_0_0_2_1 * dm_kl[0];
            vj_ij[13] += R_0_0_2_1 * dm_kl[1];
            vj_ij[4] += R_0_0_2_2 * dm_kl[0];
            vj_ij[14] += R_0_0_2_2 * dm_kl[1];
            vj_ij[5] += R_0_0_3_1 * dm_kl[0];
            vj_ij[15] += R_0_0_3_1 * dm_kl[1];
            vj_ij[6] += R_0_1_1_1 * dm_kl[0];
            vj_ij[16] += R_0_1_1_1 * dm_kl[1];
            vj_ij[7] += R_0_1_1_2 * dm_kl[0];
            vj_ij[17] += R_0_1_1_2 * dm_kl[1];
            vj_ij[8] += R_0_1_2_1 * dm_kl[0];
            vj_ij[18] += R_0_1_2_1 * dm_kl[1];
            vj_ij[9] += R_0_2_1_1 * dm_kl[0];
            vj_ij[19] += R_0_2_1_1 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+5]; }
            vj_ij[0] += R_0_0_2_0 * dm_kl[0];
            vj_ij[10] += R_0_0_2_0 * dm_kl[1];
            vj_ij[1] += R_0_0_2_1 * dm_kl[0];
            vj_ij[11] += R_0_0_2_1 * dm_kl[1];
            vj_ij[2] += R_0_0_2_2 * dm_kl[0];
            vj_ij[12] += R_0_0_2_2 * dm_kl[1];
            vj_ij[3] += R_0_0_3_0 * dm_kl[0];
            vj_ij[13] += R_0_0_3_0 * dm_kl[1];
            vj_ij[4] += R_0_0_3_1 * dm_kl[0];
            vj_ij[14] += R_0_0_3_1 * dm_kl[1];
            vj_ij[5] += R_0_0_4_0 * dm_kl[0];
            vj_ij[15] += R_0_0_4_0 * dm_kl[1];
            vj_ij[6] += R_0_1_2_0 * dm_kl[0];
            vj_ij[16] += R_0_1_2_0 * dm_kl[1];
            vj_ij[7] += R_0_1_2_1 * dm_kl[0];
            vj_ij[17] += R_0_1_2_1 * dm_kl[1];
            vj_ij[8] += R_0_1_3_0 * dm_kl[0];
            vj_ij[18] += R_0_1_3_0 * dm_kl[1];
            vj_ij[9] += R_0_2_2_0 * dm_kl[0];
            vj_ij[19] += R_0_2_2_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+6]; }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[10] += R_0_1_0_0 * dm_kl[1];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[11] += R_0_1_0_1 * dm_kl[1];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[12] += R_0_1_0_2 * dm_kl[1];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[13] += R_0_1_1_0 * dm_kl[1];
            vj_ij[4] += R_0_1_1_1 * dm_kl[0];
            vj_ij[14] += R_0_1_1_1 * dm_kl[1];
            vj_ij[5] += R_0_1_2_0 * dm_kl[0];
            vj_ij[15] += R_0_1_2_0 * dm_kl[1];
            vj_ij[6] += R_0_2_0_0 * dm_kl[0];
            vj_ij[16] += R_0_2_0_0 * dm_kl[1];
            vj_ij[7] += R_0_2_0_1 * dm_kl[0];
            vj_ij[17] += R_0_2_0_1 * dm_kl[1];
            vj_ij[8] += R_0_2_1_0 * dm_kl[0];
            vj_ij[18] += R_0_2_1_0 * dm_kl[1];
            vj_ij[9] += R_0_3_0_0 * dm_kl[0];
            vj_ij[19] += R_0_3_0_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+7]; }
            vj_ij[0] += R_0_1_0_1 * dm_kl[0];
            vj_ij[10] += R_0_1_0_1 * dm_kl[1];
            vj_ij[1] += R_0_1_0_2 * dm_kl[0];
            vj_ij[11] += R_0_1_0_2 * dm_kl[1];
            vj_ij[2] += R_0_1_0_3 * dm_kl[0];
            vj_ij[12] += R_0_1_0_3 * dm_kl[1];
            vj_ij[3] += R_0_1_1_1 * dm_kl[0];
            vj_ij[13] += R_0_1_1_1 * dm_kl[1];
            vj_ij[4] += R_0_1_1_2 * dm_kl[0];
            vj_ij[14] += R_0_1_1_2 * dm_kl[1];
            vj_ij[5] += R_0_1_2_1 * dm_kl[0];
            vj_ij[15] += R_0_1_2_1 * dm_kl[1];
            vj_ij[6] += R_0_2_0_1 * dm_kl[0];
            vj_ij[16] += R_0_2_0_1 * dm_kl[1];
            vj_ij[7] += R_0_2_0_2 * dm_kl[0];
            vj_ij[17] += R_0_2_0_2 * dm_kl[1];
            vj_ij[8] += R_0_2_1_1 * dm_kl[0];
            vj_ij[18] += R_0_2_1_1 * dm_kl[1];
            vj_ij[9] += R_0_3_0_1 * dm_kl[0];
            vj_ij[19] += R_0_3_0_1 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+8]; }
            vj_ij[0] += R_0_1_1_0 * dm_kl[0];
            vj_ij[10] += R_0_1_1_0 * dm_kl[1];
            vj_ij[1] += R_0_1_1_1 * dm_kl[0];
            vj_ij[11] += R_0_1_1_1 * dm_kl[1];
            vj_ij[2] += R_0_1_1_2 * dm_kl[0];
            vj_ij[12] += R_0_1_1_2 * dm_kl[1];
            vj_ij[3] += R_0_1_2_0 * dm_kl[0];
            vj_ij[13] += R_0_1_2_0 * dm_kl[1];
            vj_ij[4] += R_0_1_2_1 * dm_kl[0];
            vj_ij[14] += R_0_1_2_1 * dm_kl[1];
            vj_ij[5] += R_0_1_3_0 * dm_kl[0];
            vj_ij[15] += R_0_1_3_0 * dm_kl[1];
            vj_ij[6] += R_0_2_1_0 * dm_kl[0];
            vj_ij[16] += R_0_2_1_0 * dm_kl[1];
            vj_ij[7] += R_0_2_1_1 * dm_kl[0];
            vj_ij[17] += R_0_2_1_1 * dm_kl[1];
            vj_ij[8] += R_0_2_2_0 * dm_kl[0];
            vj_ij[18] += R_0_2_2_0 * dm_kl[1];
            vj_ij[9] += R_0_3_1_0 * dm_kl[0];
            vj_ij[19] += R_0_3_1_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+9]; }
            vj_ij[0] += R_0_2_0_0 * dm_kl[0];
            vj_ij[10] += R_0_2_0_0 * dm_kl[1];
            vj_ij[1] += R_0_2_0_1 * dm_kl[0];
            vj_ij[11] += R_0_2_0_1 * dm_kl[1];
            vj_ij[2] += R_0_2_0_2 * dm_kl[0];
            vj_ij[12] += R_0_2_0_2 * dm_kl[1];
            vj_ij[3] += R_0_2_1_0 * dm_kl[0];
            vj_ij[13] += R_0_2_1_0 * dm_kl[1];
            vj_ij[4] += R_0_2_1_1 * dm_kl[0];
            vj_ij[14] += R_0_2_1_1 * dm_kl[1];
            vj_ij[5] += R_0_2_2_0 * dm_kl[0];
            vj_ij[15] += R_0_2_2_0 * dm_kl[1];
            vj_ij[6] += R_0_3_0_0 * dm_kl[0];
            vj_ij[16] += R_0_3_0_0 * dm_kl[1];
            vj_ij[7] += R_0_3_0_1 * dm_kl[0];
            vj_ij[17] += R_0_3_0_1 * dm_kl[1];
            vj_ij[8] += R_0_3_1_0 * dm_kl[0];
            vj_ij[18] += R_0_3_1_0 * dm_kl[1];
            vj_ij[9] += R_0_4_0_0 * dm_kl[0];
            vj_ij[19] += R_0_4_0_0 * dm_kl[1];
            } else {
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[160];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[320];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[480];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[176];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[336];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[496];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[192];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[352];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[512];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[208];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[368];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[528];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[224];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[384];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[544];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[240];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[400];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[560];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[256];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[416];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[576];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[272];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[432];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[592];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[288];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[448];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[608];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[304];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[464];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[160];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[320];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[480];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[176];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[336];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[496];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[192];
            vj_kl[2] -= R_0_0_0_3 * dm_ij_cache[352];
            vj_kl[3] -= R_0_0_0_3 * dm_ij_cache[512];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[208];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[368];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[528];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[224];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[384];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[544];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[240];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[400];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[560];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[256];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[416];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[576];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[272];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[432];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[592];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[288];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[448];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[608];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[304];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[464];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+64] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[0];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[160];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[320];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[480];
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[176];
            vj_kl[2] += R_0_0_0_3 * dm_ij_cache[336];
            vj_kl[3] += R_0_0_0_3 * dm_ij_cache[496];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_4 * dm_ij_cache[192];
            vj_kl[2] += R_0_0_0_4 * dm_ij_cache[352];
            vj_kl[3] += R_0_0_0_4 * dm_ij_cache[512];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[208];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[368];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[528];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[224];
            vj_kl[2] += R_0_0_1_3 * dm_ij_cache[384];
            vj_kl[3] += R_0_0_1_3 * dm_ij_cache[544];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[240];
            vj_kl[2] += R_0_0_2_2 * dm_ij_cache[400];
            vj_kl[3] += R_0_0_2_2 * dm_ij_cache[560];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[256];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[416];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[576];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[272];
            vj_kl[2] += R_0_1_0_3 * dm_ij_cache[432];
            vj_kl[3] += R_0_1_0_3 * dm_ij_cache[592];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[288];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[448];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[608];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[304];
            vj_kl[2] += R_0_2_0_2 * dm_ij_cache[464];
            vj_kl[3] += R_0_2_0_2 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+128] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[160];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[320];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[176];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[336];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[496];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[192];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[352];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[512];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[208];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[368];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[528];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[224];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[384];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[544];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[240];
            vj_kl[2] -= R_0_0_3_0 * dm_ij_cache[400];
            vj_kl[3] -= R_0_0_3_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[256];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[416];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[576];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[272];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[432];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[592];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[288];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[448];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[608];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[304];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[464];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+192] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[0];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[160];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[320];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[480];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[176];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[336];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[496];
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[192];
            vj_kl[2] += R_0_0_1_3 * dm_ij_cache[352];
            vj_kl[3] += R_0_0_1_3 * dm_ij_cache[512];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[208];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[368];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[528];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[224];
            vj_kl[2] += R_0_0_2_2 * dm_ij_cache[384];
            vj_kl[3] += R_0_0_2_2 * dm_ij_cache[544];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[240];
            vj_kl[2] += R_0_0_3_1 * dm_ij_cache[400];
            vj_kl[3] += R_0_0_3_1 * dm_ij_cache[560];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[256];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[416];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[576];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[272];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[432];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[592];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[288];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[448];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[608];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[304];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[464];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+256] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[0];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[160];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[320];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[480];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[176];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[336];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[496];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[192];
            vj_kl[2] += R_0_0_2_2 * dm_ij_cache[352];
            vj_kl[3] += R_0_0_2_2 * dm_ij_cache[512];
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[208];
            vj_kl[2] += R_0_0_3_0 * dm_ij_cache[368];
            vj_kl[3] += R_0_0_3_0 * dm_ij_cache[528];
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[224];
            vj_kl[2] += R_0_0_3_1 * dm_ij_cache[384];
            vj_kl[3] += R_0_0_3_1 * dm_ij_cache[544];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_4_0 * dm_ij_cache[240];
            vj_kl[2] += R_0_0_4_0 * dm_ij_cache[400];
            vj_kl[3] += R_0_0_4_0 * dm_ij_cache[560];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[256];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[416];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[576];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[272];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[432];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[592];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[288];
            vj_kl[2] += R_0_1_3_0 * dm_ij_cache[448];
            vj_kl[3] += R_0_1_3_0 * dm_ij_cache[608];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[304];
            vj_kl[2] += R_0_2_2_0 * dm_ij_cache[464];
            vj_kl[3] += R_0_2_2_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+320] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[320];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[336];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[496];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[352];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[512];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[368];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[528];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[384];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[544];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[400];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[416];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[576];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[432];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[592];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[288];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[448];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[608];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[304];
            vj_kl[2] -= R_0_3_0_0 * dm_ij_cache[464];
            vj_kl[3] -= R_0_3_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+384] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[0];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[160];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[320];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[480];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[16];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[176];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[336];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[496];
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[32];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[192];
            vj_kl[2] += R_0_1_0_3 * dm_ij_cache[352];
            vj_kl[3] += R_0_1_0_3 * dm_ij_cache[512];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[208];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[368];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[528];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[64];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[224];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[384];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[544];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[240];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[400];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[560];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[96];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[256];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[416];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[576];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[112];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[272];
            vj_kl[2] += R_0_2_0_2 * dm_ij_cache[432];
            vj_kl[3] += R_0_2_0_2 * dm_ij_cache[592];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[288];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[448];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[608];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[144];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[304];
            vj_kl[2] += R_0_3_0_1 * dm_ij_cache[464];
            vj_kl[3] += R_0_3_0_1 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+448] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[0];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[160];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[320];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[480];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[176];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[336];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[496];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[192];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[352];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[512];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[208];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[368];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[528];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[224];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[384];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[544];
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[240];
            vj_kl[2] += R_0_1_3_0 * dm_ij_cache[400];
            vj_kl[3] += R_0_1_3_0 * dm_ij_cache[560];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[256];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[416];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[576];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[272];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[432];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[592];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[288];
            vj_kl[2] += R_0_2_2_0 * dm_ij_cache[448];
            vj_kl[3] += R_0_2_2_0 * dm_ij_cache[608];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[304];
            vj_kl[2] += R_0_3_1_0 * dm_ij_cache[464];
            vj_kl[3] += R_0_3_1_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+512] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[0];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[160];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[320];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[480];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[176];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[336];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[496];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[192];
            vj_kl[2] += R_0_2_0_2 * dm_ij_cache[352];
            vj_kl[3] += R_0_2_0_2 * dm_ij_cache[512];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[48];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[208];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[368];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[528];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[64];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[224];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[384];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[544];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[240];
            vj_kl[2] += R_0_2_2_0 * dm_ij_cache[400];
            vj_kl[3] += R_0_2_2_0 * dm_ij_cache[560];
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[256];
            vj_kl[2] += R_0_3_0_0 * dm_ij_cache[416];
            vj_kl[3] += R_0_3_0_0 * dm_ij_cache[576];
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[272];
            vj_kl[2] += R_0_3_0_1 * dm_ij_cache[432];
            vj_kl[3] += R_0_3_0_1 * dm_ij_cache[592];
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[128];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[288];
            vj_kl[2] += R_0_3_1_0 * dm_ij_cache[448];
            vj_kl[3] += R_0_3_1_0 * dm_ij_cache[608];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_4_0_0 * dm_ij_cache[304];
            vj_kl[2] += R_0_4_0_0 * dm_ij_cache[464];
            vj_kl[3] += R_0_4_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*640+576] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[30] += gamma_inc[0*256] * dm_kl[3];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[11] += R_0_0_0_1 * dm_kl[1];
            vj_ij[21] += R_0_0_0_1 * dm_kl[2];
            vj_ij[31] += R_0_0_0_1 * dm_kl[3];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[12] += R_0_0_0_2 * dm_kl[1];
            vj_ij[22] += R_0_0_0_2 * dm_kl[2];
            vj_ij[32] += R_0_0_0_2 * dm_kl[3];
            vj_ij[3] += R_0_0_1_0 * dm_kl[0];
            vj_ij[13] += R_0_0_1_0 * dm_kl[1];
            vj_ij[23] += R_0_0_1_0 * dm_kl[2];
            vj_ij[33] += R_0_0_1_0 * dm_kl[3];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[14] += R_0_0_1_1 * dm_kl[1];
            vj_ij[24] += R_0_0_1_1 * dm_kl[2];
            vj_ij[34] += R_0_0_1_1 * dm_kl[3];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[15] += R_0_0_2_0 * dm_kl[1];
            vj_ij[25] += R_0_0_2_0 * dm_kl[2];
            vj_ij[35] += R_0_0_2_0 * dm_kl[3];
            vj_ij[6] += R_0_1_0_0 * dm_kl[0];
            vj_ij[16] += R_0_1_0_0 * dm_kl[1];
            vj_ij[26] += R_0_1_0_0 * dm_kl[2];
            vj_ij[36] += R_0_1_0_0 * dm_kl[3];
            vj_ij[7] += R_0_1_0_1 * dm_kl[0];
            vj_ij[17] += R_0_1_0_1 * dm_kl[1];
            vj_ij[27] += R_0_1_0_1 * dm_kl[2];
            vj_ij[37] += R_0_1_0_1 * dm_kl[3];
            vj_ij[8] += R_0_1_1_0 * dm_kl[0];
            vj_ij[18] += R_0_1_1_0 * dm_kl[1];
            vj_ij[28] += R_0_1_1_0 * dm_kl[2];
            vj_ij[38] += R_0_1_1_0 * dm_kl[3];
            vj_ij[9] += R_0_2_0_0 * dm_kl[0];
            vj_ij[19] += R_0_2_0_0 * dm_kl[1];
            vj_ij[29] += R_0_2_0_0 * dm_kl[2];
            vj_ij[39] += R_0_2_0_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1];
            }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[10] += R_0_0_0_1 * dm_kl[1];
            vj_ij[20] += R_0_0_0_1 * dm_kl[2];
            vj_ij[30] += R_0_0_0_1 * dm_kl[3];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[11] += R_0_0_0_2 * dm_kl[1];
            vj_ij[21] += R_0_0_0_2 * dm_kl[2];
            vj_ij[31] += R_0_0_0_2 * dm_kl[3];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[12] += R_0_0_0_3 * dm_kl[1];
            vj_ij[22] += R_0_0_0_3 * dm_kl[2];
            vj_ij[32] += R_0_0_0_3 * dm_kl[3];
            vj_ij[3] += R_0_0_1_1 * dm_kl[0];
            vj_ij[13] += R_0_0_1_1 * dm_kl[1];
            vj_ij[23] += R_0_0_1_1 * dm_kl[2];
            vj_ij[33] += R_0_0_1_1 * dm_kl[3];
            vj_ij[4] += R_0_0_1_2 * dm_kl[0];
            vj_ij[14] += R_0_0_1_2 * dm_kl[1];
            vj_ij[24] += R_0_0_1_2 * dm_kl[2];
            vj_ij[34] += R_0_0_1_2 * dm_kl[3];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[15] += R_0_0_2_1 * dm_kl[1];
            vj_ij[25] += R_0_0_2_1 * dm_kl[2];
            vj_ij[35] += R_0_0_2_1 * dm_kl[3];
            vj_ij[6] += R_0_1_0_1 * dm_kl[0];
            vj_ij[16] += R_0_1_0_1 * dm_kl[1];
            vj_ij[26] += R_0_1_0_1 * dm_kl[2];
            vj_ij[36] += R_0_1_0_1 * dm_kl[3];
            vj_ij[7] += R_0_1_0_2 * dm_kl[0];
            vj_ij[17] += R_0_1_0_2 * dm_kl[1];
            vj_ij[27] += R_0_1_0_2 * dm_kl[2];
            vj_ij[37] += R_0_1_0_2 * dm_kl[3];
            vj_ij[8] += R_0_1_1_1 * dm_kl[0];
            vj_ij[18] += R_0_1_1_1 * dm_kl[1];
            vj_ij[28] += R_0_1_1_1 * dm_kl[2];
            vj_ij[38] += R_0_1_1_1 * dm_kl[3];
            vj_ij[9] += R_0_2_0_1 * dm_kl[0];
            vj_ij[19] += R_0_2_0_1 * dm_kl[1];
            vj_ij[29] += R_0_2_0_1 * dm_kl[2];
            vj_ij[39] += R_0_2_0_1 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+2];
            }
            vj_ij[0] += R_0_0_0_2 * dm_kl[0];
            vj_ij[10] += R_0_0_0_2 * dm_kl[1];
            vj_ij[20] += R_0_0_0_2 * dm_kl[2];
            vj_ij[30] += R_0_0_0_2 * dm_kl[3];
            vj_ij[1] += R_0_0_0_3 * dm_kl[0];
            vj_ij[11] += R_0_0_0_3 * dm_kl[1];
            vj_ij[21] += R_0_0_0_3 * dm_kl[2];
            vj_ij[31] += R_0_0_0_3 * dm_kl[3];
            vj_ij[2] += R_0_0_0_4 * dm_kl[0];
            vj_ij[12] += R_0_0_0_4 * dm_kl[1];
            vj_ij[22] += R_0_0_0_4 * dm_kl[2];
            vj_ij[32] += R_0_0_0_4 * dm_kl[3];
            vj_ij[3] += R_0_0_1_2 * dm_kl[0];
            vj_ij[13] += R_0_0_1_2 * dm_kl[1];
            vj_ij[23] += R_0_0_1_2 * dm_kl[2];
            vj_ij[33] += R_0_0_1_2 * dm_kl[3];
            vj_ij[4] += R_0_0_1_3 * dm_kl[0];
            vj_ij[14] += R_0_0_1_3 * dm_kl[1];
            vj_ij[24] += R_0_0_1_3 * dm_kl[2];
            vj_ij[34] += R_0_0_1_3 * dm_kl[3];
            vj_ij[5] += R_0_0_2_2 * dm_kl[0];
            vj_ij[15] += R_0_0_2_2 * dm_kl[1];
            vj_ij[25] += R_0_0_2_2 * dm_kl[2];
            vj_ij[35] += R_0_0_2_2 * dm_kl[3];
            vj_ij[6] += R_0_1_0_2 * dm_kl[0];
            vj_ij[16] += R_0_1_0_2 * dm_kl[1];
            vj_ij[26] += R_0_1_0_2 * dm_kl[2];
            vj_ij[36] += R_0_1_0_2 * dm_kl[3];
            vj_ij[7] += R_0_1_0_3 * dm_kl[0];
            vj_ij[17] += R_0_1_0_3 * dm_kl[1];
            vj_ij[27] += R_0_1_0_3 * dm_kl[2];
            vj_ij[37] += R_0_1_0_3 * dm_kl[3];
            vj_ij[8] += R_0_1_1_2 * dm_kl[0];
            vj_ij[18] += R_0_1_1_2 * dm_kl[1];
            vj_ij[28] += R_0_1_1_2 * dm_kl[2];
            vj_ij[38] += R_0_1_1_2 * dm_kl[3];
            vj_ij[9] += R_0_2_0_2 * dm_kl[0];
            vj_ij[19] += R_0_2_0_2 * dm_kl[1];
            vj_ij[29] += R_0_2_0_2 * dm_kl[2];
            vj_ij[39] += R_0_2_0_2 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3];
            }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[10] += R_0_0_1_0 * dm_kl[1];
            vj_ij[20] += R_0_0_1_0 * dm_kl[2];
            vj_ij[30] += R_0_0_1_0 * dm_kl[3];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[11] += R_0_0_1_1 * dm_kl[1];
            vj_ij[21] += R_0_0_1_1 * dm_kl[2];
            vj_ij[31] += R_0_0_1_1 * dm_kl[3];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[12] += R_0_0_1_2 * dm_kl[1];
            vj_ij[22] += R_0_0_1_2 * dm_kl[2];
            vj_ij[32] += R_0_0_1_2 * dm_kl[3];
            vj_ij[3] += R_0_0_2_0 * dm_kl[0];
            vj_ij[13] += R_0_0_2_0 * dm_kl[1];
            vj_ij[23] += R_0_0_2_0 * dm_kl[2];
            vj_ij[33] += R_0_0_2_0 * dm_kl[3];
            vj_ij[4] += R_0_0_2_1 * dm_kl[0];
            vj_ij[14] += R_0_0_2_1 * dm_kl[1];
            vj_ij[24] += R_0_0_2_1 * dm_kl[2];
            vj_ij[34] += R_0_0_2_1 * dm_kl[3];
            vj_ij[5] += R_0_0_3_0 * dm_kl[0];
            vj_ij[15] += R_0_0_3_0 * dm_kl[1];
            vj_ij[25] += R_0_0_3_0 * dm_kl[2];
            vj_ij[35] += R_0_0_3_0 * dm_kl[3];
            vj_ij[6] += R_0_1_1_0 * dm_kl[0];
            vj_ij[16] += R_0_1_1_0 * dm_kl[1];
            vj_ij[26] += R_0_1_1_0 * dm_kl[2];
            vj_ij[36] += R_0_1_1_0 * dm_kl[3];
            vj_ij[7] += R_0_1_1_1 * dm_kl[0];
            vj_ij[17] += R_0_1_1_1 * dm_kl[1];
            vj_ij[27] += R_0_1_1_1 * dm_kl[2];
            vj_ij[37] += R_0_1_1_1 * dm_kl[3];
            vj_ij[8] += R_0_1_2_0 * dm_kl[0];
            vj_ij[18] += R_0_1_2_0 * dm_kl[1];
            vj_ij[28] += R_0_1_2_0 * dm_kl[2];
            vj_ij[38] += R_0_1_2_0 * dm_kl[3];
            vj_ij[9] += R_0_2_1_0 * dm_kl[0];
            vj_ij[19] += R_0_2_1_0 * dm_kl[1];
            vj_ij[29] += R_0_2_1_0 * dm_kl[2];
            vj_ij[39] += R_0_2_1_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+4];
            }
            vj_ij[0] += R_0_0_1_1 * dm_kl[0];
            vj_ij[10] += R_0_0_1_1 * dm_kl[1];
            vj_ij[20] += R_0_0_1_1 * dm_kl[2];
            vj_ij[30] += R_0_0_1_1 * dm_kl[3];
            vj_ij[1] += R_0_0_1_2 * dm_kl[0];
            vj_ij[11] += R_0_0_1_2 * dm_kl[1];
            vj_ij[21] += R_0_0_1_2 * dm_kl[2];
            vj_ij[31] += R_0_0_1_2 * dm_kl[3];
            vj_ij[2] += R_0_0_1_3 * dm_kl[0];
            vj_ij[12] += R_0_0_1_3 * dm_kl[1];
            vj_ij[22] += R_0_0_1_3 * dm_kl[2];
            vj_ij[32] += R_0_0_1_3 * dm_kl[3];
            vj_ij[3] += R_0_0_2_1 * dm_kl[0];
            vj_ij[13] += R_0_0_2_1 * dm_kl[1];
            vj_ij[23] += R_0_0_2_1 * dm_kl[2];
            vj_ij[33] += R_0_0_2_1 * dm_kl[3];
            vj_ij[4] += R_0_0_2_2 * dm_kl[0];
            vj_ij[14] += R_0_0_2_2 * dm_kl[1];
            vj_ij[24] += R_0_0_2_2 * dm_kl[2];
            vj_ij[34] += R_0_0_2_2 * dm_kl[3];
            vj_ij[5] += R_0_0_3_1 * dm_kl[0];
            vj_ij[15] += R_0_0_3_1 * dm_kl[1];
            vj_ij[25] += R_0_0_3_1 * dm_kl[2];
            vj_ij[35] += R_0_0_3_1 * dm_kl[3];
            vj_ij[6] += R_0_1_1_1 * dm_kl[0];
            vj_ij[16] += R_0_1_1_1 * dm_kl[1];
            vj_ij[26] += R_0_1_1_1 * dm_kl[2];
            vj_ij[36] += R_0_1_1_1 * dm_kl[3];
            vj_ij[7] += R_0_1_1_2 * dm_kl[0];
            vj_ij[17] += R_0_1_1_2 * dm_kl[1];
            vj_ij[27] += R_0_1_1_2 * dm_kl[2];
            vj_ij[37] += R_0_1_1_2 * dm_kl[3];
            vj_ij[8] += R_0_1_2_1 * dm_kl[0];
            vj_ij[18] += R_0_1_2_1 * dm_kl[1];
            vj_ij[28] += R_0_1_2_1 * dm_kl[2];
            vj_ij[38] += R_0_1_2_1 * dm_kl[3];
            vj_ij[9] += R_0_2_1_1 * dm_kl[0];
            vj_ij[19] += R_0_2_1_1 * dm_kl[1];
            vj_ij[29] += R_0_2_1_1 * dm_kl[2];
            vj_ij[39] += R_0_2_1_1 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+5];
            }
            vj_ij[0] += R_0_0_2_0 * dm_kl[0];
            vj_ij[10] += R_0_0_2_0 * dm_kl[1];
            vj_ij[20] += R_0_0_2_0 * dm_kl[2];
            vj_ij[30] += R_0_0_2_0 * dm_kl[3];
            vj_ij[1] += R_0_0_2_1 * dm_kl[0];
            vj_ij[11] += R_0_0_2_1 * dm_kl[1];
            vj_ij[21] += R_0_0_2_1 * dm_kl[2];
            vj_ij[31] += R_0_0_2_1 * dm_kl[3];
            vj_ij[2] += R_0_0_2_2 * dm_kl[0];
            vj_ij[12] += R_0_0_2_2 * dm_kl[1];
            vj_ij[22] += R_0_0_2_2 * dm_kl[2];
            vj_ij[32] += R_0_0_2_2 * dm_kl[3];
            vj_ij[3] += R_0_0_3_0 * dm_kl[0];
            vj_ij[13] += R_0_0_3_0 * dm_kl[1];
            vj_ij[23] += R_0_0_3_0 * dm_kl[2];
            vj_ij[33] += R_0_0_3_0 * dm_kl[3];
            vj_ij[4] += R_0_0_3_1 * dm_kl[0];
            vj_ij[14] += R_0_0_3_1 * dm_kl[1];
            vj_ij[24] += R_0_0_3_1 * dm_kl[2];
            vj_ij[34] += R_0_0_3_1 * dm_kl[3];
            vj_ij[5] += R_0_0_4_0 * dm_kl[0];
            vj_ij[15] += R_0_0_4_0 * dm_kl[1];
            vj_ij[25] += R_0_0_4_0 * dm_kl[2];
            vj_ij[35] += R_0_0_4_0 * dm_kl[3];
            vj_ij[6] += R_0_1_2_0 * dm_kl[0];
            vj_ij[16] += R_0_1_2_0 * dm_kl[1];
            vj_ij[26] += R_0_1_2_0 * dm_kl[2];
            vj_ij[36] += R_0_1_2_0 * dm_kl[3];
            vj_ij[7] += R_0_1_2_1 * dm_kl[0];
            vj_ij[17] += R_0_1_2_1 * dm_kl[1];
            vj_ij[27] += R_0_1_2_1 * dm_kl[2];
            vj_ij[37] += R_0_1_2_1 * dm_kl[3];
            vj_ij[8] += R_0_1_3_0 * dm_kl[0];
            vj_ij[18] += R_0_1_3_0 * dm_kl[1];
            vj_ij[28] += R_0_1_3_0 * dm_kl[2];
            vj_ij[38] += R_0_1_3_0 * dm_kl[3];
            vj_ij[9] += R_0_2_2_0 * dm_kl[0];
            vj_ij[19] += R_0_2_2_0 * dm_kl[1];
            vj_ij[29] += R_0_2_2_0 * dm_kl[2];
            vj_ij[39] += R_0_2_2_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+6];
            }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[10] += R_0_1_0_0 * dm_kl[1];
            vj_ij[20] += R_0_1_0_0 * dm_kl[2];
            vj_ij[30] += R_0_1_0_0 * dm_kl[3];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[11] += R_0_1_0_1 * dm_kl[1];
            vj_ij[21] += R_0_1_0_1 * dm_kl[2];
            vj_ij[31] += R_0_1_0_1 * dm_kl[3];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[12] += R_0_1_0_2 * dm_kl[1];
            vj_ij[22] += R_0_1_0_2 * dm_kl[2];
            vj_ij[32] += R_0_1_0_2 * dm_kl[3];
            vj_ij[3] += R_0_1_1_0 * dm_kl[0];
            vj_ij[13] += R_0_1_1_0 * dm_kl[1];
            vj_ij[23] += R_0_1_1_0 * dm_kl[2];
            vj_ij[33] += R_0_1_1_0 * dm_kl[3];
            vj_ij[4] += R_0_1_1_1 * dm_kl[0];
            vj_ij[14] += R_0_1_1_1 * dm_kl[1];
            vj_ij[24] += R_0_1_1_1 * dm_kl[2];
            vj_ij[34] += R_0_1_1_1 * dm_kl[3];
            vj_ij[5] += R_0_1_2_0 * dm_kl[0];
            vj_ij[15] += R_0_1_2_0 * dm_kl[1];
            vj_ij[25] += R_0_1_2_0 * dm_kl[2];
            vj_ij[35] += R_0_1_2_0 * dm_kl[3];
            vj_ij[6] += R_0_2_0_0 * dm_kl[0];
            vj_ij[16] += R_0_2_0_0 * dm_kl[1];
            vj_ij[26] += R_0_2_0_0 * dm_kl[2];
            vj_ij[36] += R_0_2_0_0 * dm_kl[3];
            vj_ij[7] += R_0_2_0_1 * dm_kl[0];
            vj_ij[17] += R_0_2_0_1 * dm_kl[1];
            vj_ij[27] += R_0_2_0_1 * dm_kl[2];
            vj_ij[37] += R_0_2_0_1 * dm_kl[3];
            vj_ij[8] += R_0_2_1_0 * dm_kl[0];
            vj_ij[18] += R_0_2_1_0 * dm_kl[1];
            vj_ij[28] += R_0_2_1_0 * dm_kl[2];
            vj_ij[38] += R_0_2_1_0 * dm_kl[3];
            vj_ij[9] += R_0_3_0_0 * dm_kl[0];
            vj_ij[19] += R_0_3_0_0 * dm_kl[1];
            vj_ij[29] += R_0_3_0_0 * dm_kl[2];
            vj_ij[39] += R_0_3_0_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+7];
            }
            vj_ij[0] += R_0_1_0_1 * dm_kl[0];
            vj_ij[10] += R_0_1_0_1 * dm_kl[1];
            vj_ij[20] += R_0_1_0_1 * dm_kl[2];
            vj_ij[30] += R_0_1_0_1 * dm_kl[3];
            vj_ij[1] += R_0_1_0_2 * dm_kl[0];
            vj_ij[11] += R_0_1_0_2 * dm_kl[1];
            vj_ij[21] += R_0_1_0_2 * dm_kl[2];
            vj_ij[31] += R_0_1_0_2 * dm_kl[3];
            vj_ij[2] += R_0_1_0_3 * dm_kl[0];
            vj_ij[12] += R_0_1_0_3 * dm_kl[1];
            vj_ij[22] += R_0_1_0_3 * dm_kl[2];
            vj_ij[32] += R_0_1_0_3 * dm_kl[3];
            vj_ij[3] += R_0_1_1_1 * dm_kl[0];
            vj_ij[13] += R_0_1_1_1 * dm_kl[1];
            vj_ij[23] += R_0_1_1_1 * dm_kl[2];
            vj_ij[33] += R_0_1_1_1 * dm_kl[3];
            vj_ij[4] += R_0_1_1_2 * dm_kl[0];
            vj_ij[14] += R_0_1_1_2 * dm_kl[1];
            vj_ij[24] += R_0_1_1_2 * dm_kl[2];
            vj_ij[34] += R_0_1_1_2 * dm_kl[3];
            vj_ij[5] += R_0_1_2_1 * dm_kl[0];
            vj_ij[15] += R_0_1_2_1 * dm_kl[1];
            vj_ij[25] += R_0_1_2_1 * dm_kl[2];
            vj_ij[35] += R_0_1_2_1 * dm_kl[3];
            vj_ij[6] += R_0_2_0_1 * dm_kl[0];
            vj_ij[16] += R_0_2_0_1 * dm_kl[1];
            vj_ij[26] += R_0_2_0_1 * dm_kl[2];
            vj_ij[36] += R_0_2_0_1 * dm_kl[3];
            vj_ij[7] += R_0_2_0_2 * dm_kl[0];
            vj_ij[17] += R_0_2_0_2 * dm_kl[1];
            vj_ij[27] += R_0_2_0_2 * dm_kl[2];
            vj_ij[37] += R_0_2_0_2 * dm_kl[3];
            vj_ij[8] += R_0_2_1_1 * dm_kl[0];
            vj_ij[18] += R_0_2_1_1 * dm_kl[1];
            vj_ij[28] += R_0_2_1_1 * dm_kl[2];
            vj_ij[38] += R_0_2_1_1 * dm_kl[3];
            vj_ij[9] += R_0_3_0_1 * dm_kl[0];
            vj_ij[19] += R_0_3_0_1 * dm_kl[1];
            vj_ij[29] += R_0_3_0_1 * dm_kl[2];
            vj_ij[39] += R_0_3_0_1 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+8];
            }
            vj_ij[0] += R_0_1_1_0 * dm_kl[0];
            vj_ij[10] += R_0_1_1_0 * dm_kl[1];
            vj_ij[20] += R_0_1_1_0 * dm_kl[2];
            vj_ij[30] += R_0_1_1_0 * dm_kl[3];
            vj_ij[1] += R_0_1_1_1 * dm_kl[0];
            vj_ij[11] += R_0_1_1_1 * dm_kl[1];
            vj_ij[21] += R_0_1_1_1 * dm_kl[2];
            vj_ij[31] += R_0_1_1_1 * dm_kl[3];
            vj_ij[2] += R_0_1_1_2 * dm_kl[0];
            vj_ij[12] += R_0_1_1_2 * dm_kl[1];
            vj_ij[22] += R_0_1_1_2 * dm_kl[2];
            vj_ij[32] += R_0_1_1_2 * dm_kl[3];
            vj_ij[3] += R_0_1_2_0 * dm_kl[0];
            vj_ij[13] += R_0_1_2_0 * dm_kl[1];
            vj_ij[23] += R_0_1_2_0 * dm_kl[2];
            vj_ij[33] += R_0_1_2_0 * dm_kl[3];
            vj_ij[4] += R_0_1_2_1 * dm_kl[0];
            vj_ij[14] += R_0_1_2_1 * dm_kl[1];
            vj_ij[24] += R_0_1_2_1 * dm_kl[2];
            vj_ij[34] += R_0_1_2_1 * dm_kl[3];
            vj_ij[5] += R_0_1_3_0 * dm_kl[0];
            vj_ij[15] += R_0_1_3_0 * dm_kl[1];
            vj_ij[25] += R_0_1_3_0 * dm_kl[2];
            vj_ij[35] += R_0_1_3_0 * dm_kl[3];
            vj_ij[6] += R_0_2_1_0 * dm_kl[0];
            vj_ij[16] += R_0_2_1_0 * dm_kl[1];
            vj_ij[26] += R_0_2_1_0 * dm_kl[2];
            vj_ij[36] += R_0_2_1_0 * dm_kl[3];
            vj_ij[7] += R_0_2_1_1 * dm_kl[0];
            vj_ij[17] += R_0_2_1_1 * dm_kl[1];
            vj_ij[27] += R_0_2_1_1 * dm_kl[2];
            vj_ij[37] += R_0_2_1_1 * dm_kl[3];
            vj_ij[8] += R_0_2_2_0 * dm_kl[0];
            vj_ij[18] += R_0_2_2_0 * dm_kl[1];
            vj_ij[28] += R_0_2_2_0 * dm_kl[2];
            vj_ij[38] += R_0_2_2_0 * dm_kl[3];
            vj_ij[9] += R_0_3_1_0 * dm_kl[0];
            vj_ij[19] += R_0_3_1_0 * dm_kl[1];
            vj_ij[29] += R_0_3_1_0 * dm_kl[2];
            vj_ij[39] += R_0_3_1_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+9];
            }
            vj_ij[0] += R_0_2_0_0 * dm_kl[0];
            vj_ij[10] += R_0_2_0_0 * dm_kl[1];
            vj_ij[20] += R_0_2_0_0 * dm_kl[2];
            vj_ij[30] += R_0_2_0_0 * dm_kl[3];
            vj_ij[1] += R_0_2_0_1 * dm_kl[0];
            vj_ij[11] += R_0_2_0_1 * dm_kl[1];
            vj_ij[21] += R_0_2_0_1 * dm_kl[2];
            vj_ij[31] += R_0_2_0_1 * dm_kl[3];
            vj_ij[2] += R_0_2_0_2 * dm_kl[0];
            vj_ij[12] += R_0_2_0_2 * dm_kl[1];
            vj_ij[22] += R_0_2_0_2 * dm_kl[2];
            vj_ij[32] += R_0_2_0_2 * dm_kl[3];
            vj_ij[3] += R_0_2_1_0 * dm_kl[0];
            vj_ij[13] += R_0_2_1_0 * dm_kl[1];
            vj_ij[23] += R_0_2_1_0 * dm_kl[2];
            vj_ij[33] += R_0_2_1_0 * dm_kl[3];
            vj_ij[4] += R_0_2_1_1 * dm_kl[0];
            vj_ij[14] += R_0_2_1_1 * dm_kl[1];
            vj_ij[24] += R_0_2_1_1 * dm_kl[2];
            vj_ij[34] += R_0_2_1_1 * dm_kl[3];
            vj_ij[5] += R_0_2_2_0 * dm_kl[0];
            vj_ij[15] += R_0_2_2_0 * dm_kl[1];
            vj_ij[25] += R_0_2_2_0 * dm_kl[2];
            vj_ij[35] += R_0_2_2_0 * dm_kl[3];
            vj_ij[6] += R_0_3_0_0 * dm_kl[0];
            vj_ij[16] += R_0_3_0_0 * dm_kl[1];
            vj_ij[26] += R_0_3_0_0 * dm_kl[2];
            vj_ij[36] += R_0_3_0_0 * dm_kl[3];
            vj_ij[7] += R_0_3_0_1 * dm_kl[0];
            vj_ij[17] += R_0_3_0_1 * dm_kl[1];
            vj_ij[27] += R_0_3_0_1 * dm_kl[2];
            vj_ij[37] += R_0_3_0_1 * dm_kl[3];
            vj_ij[8] += R_0_3_1_0 * dm_kl[0];
            vj_ij[18] += R_0_3_1_0 * dm_kl[1];
            vj_ij[28] += R_0_3_1_0 * dm_kl[2];
            vj_ij[38] += R_0_3_1_0 * dm_kl[3];
            vj_ij[9] += R_0_4_0_0 * dm_kl[0];
            vj_ij[19] += R_0_4_0_0 * dm_kl[1];
            vj_ij[29] += R_0_4_0_0 * dm_kl[2];
            vj_ij[39] += R_0_4_0_0 * dm_kl[3];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 10; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+10];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 10; ++n) {
                __syncthreads();
                for (int m = 0; m < 4; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+10*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 4; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 40; n += 16) {
        int kl = n / 4;
        int batch_kl = n - kl * 4;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 64 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*640+kl*64]);
            }
        }
    }
} }

// TILEX=48, TILEY=21
__global__ static
void md_j_4dm_3_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 336;
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
    double vj_kl[4];
    double dm_kl[4];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1344;
    double *Rp_cache = Rq_cache + 1344;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 1344 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 1408; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 336; n += 256) {
        int task_kl = blockIdx.y * 336 + n;
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
            Rq_cache[n+336] = ykl;
            Rq_cache[n+672] = zkl;
            Rq_cache[n+1008] = akl;
        } else {
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+336] = 1e5;
            Rq_cache[n+672] = 1e5;
            Rq_cache[n+1008] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 1344; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 20 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 20;
            int i = n - i_dm * 20;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[80];
        for (int ij = 0; ij < 80; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 21; ++batch_kl) {
            int task_kl0 = blockIdx.y * 336 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*21] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+336];
            double zkl = Rq_cache[sq_kl+672];
            double akl = Rq_cache[sq_kl+1008];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 3, 0, 256);
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[96];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[128];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[144];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[240];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[288];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[10] += R_0_1_0_0 * dm_kl[0];
            vj_ij[11] += R_0_1_0_1 * dm_kl[0];
            vj_ij[12] += R_0_1_0_2 * dm_kl[0];
            vj_ij[13] += R_0_1_1_0 * dm_kl[0];
            vj_ij[14] += R_0_1_1_1 * dm_kl[0];
            vj_ij[15] += R_0_1_2_0 * dm_kl[0];
            vj_ij[16] += R_0_2_0_0 * dm_kl[0];
            vj_ij[17] += R_0_2_0_1 * dm_kl[0];
            vj_ij[18] += R_0_2_1_0 * dm_kl[0];
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[320];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[336];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[352];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[368];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[384];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[400];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[416];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[432];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[448];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[464];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[480];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[496];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[512];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[528];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[544];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[560];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[576];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[592];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[608];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[21] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[22] += R_0_0_0_2 * dm_kl[1];
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[23] += R_0_0_0_3 * dm_kl[1];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            vj_ij[24] += R_0_0_1_0 * dm_kl[1];
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[25] += R_0_0_1_1 * dm_kl[1];
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[26] += R_0_0_1_2 * dm_kl[1];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            vj_ij[27] += R_0_0_2_0 * dm_kl[1];
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            vj_ij[28] += R_0_0_2_1 * dm_kl[1];
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[29] += R_0_0_3_0 * dm_kl[1];
            vj_ij[10] += R_0_1_0_0 * dm_kl[0];
            vj_ij[30] += R_0_1_0_0 * dm_kl[1];
            vj_ij[11] += R_0_1_0_1 * dm_kl[0];
            vj_ij[31] += R_0_1_0_1 * dm_kl[1];
            vj_ij[12] += R_0_1_0_2 * dm_kl[0];
            vj_ij[32] += R_0_1_0_2 * dm_kl[1];
            vj_ij[13] += R_0_1_1_0 * dm_kl[0];
            vj_ij[33] += R_0_1_1_0 * dm_kl[1];
            vj_ij[14] += R_0_1_1_1 * dm_kl[0];
            vj_ij[34] += R_0_1_1_1 * dm_kl[1];
            vj_ij[15] += R_0_1_2_0 * dm_kl[0];
            vj_ij[35] += R_0_1_2_0 * dm_kl[1];
            vj_ij[16] += R_0_2_0_0 * dm_kl[0];
            vj_ij[36] += R_0_2_0_0 * dm_kl[1];
            vj_ij[17] += R_0_2_0_1 * dm_kl[0];
            vj_ij[37] += R_0_2_0_1 * dm_kl[1];
            vj_ij[18] += R_0_2_1_0 * dm_kl[0];
            vj_ij[38] += R_0_2_1_0 * dm_kl[1];
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            vj_ij[39] += R_0_3_0_0 * dm_kl[1];
            } else {
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[320];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[640];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[960];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[336];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[656];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[976];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[352];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[672];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[992];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[368];
            vj_kl[2] += R_0_0_0_3 * dm_ij_cache[688];
            vj_kl[3] += R_0_0_0_3 * dm_ij_cache[1008];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[384];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[704];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[1024];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[400];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[720];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[1040];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[416];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[736];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[1056];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[432];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[752];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[1072];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[448];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[768];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[1088];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[464];
            vj_kl[2] += R_0_0_3_0 * dm_ij_cache[784];
            vj_kl[3] += R_0_0_3_0 * dm_ij_cache[1104];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[480];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[800];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[1120];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[496];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[816];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[1136];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[512];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[832];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[1152];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[528];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[848];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[1168];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[544];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[864];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[1184];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[560];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[880];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[1200];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[576];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[896];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[1216];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[592];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[912];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[1232];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[608];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[928];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[1248];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[624];
            vj_kl[2] += R_0_3_0_0 * dm_ij_cache[944];
            vj_kl[3] += R_0_3_0_0 * dm_ij_cache[1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[40] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[60] += gamma_inc[0*256] * dm_kl[3];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[21] += R_0_0_0_1 * dm_kl[1];
            vj_ij[41] += R_0_0_0_1 * dm_kl[2];
            vj_ij[61] += R_0_0_0_1 * dm_kl[3];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[22] += R_0_0_0_2 * dm_kl[1];
            vj_ij[42] += R_0_0_0_2 * dm_kl[2];
            vj_ij[62] += R_0_0_0_2 * dm_kl[3];
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[23] += R_0_0_0_3 * dm_kl[1];
            vj_ij[43] += R_0_0_0_3 * dm_kl[2];
            vj_ij[63] += R_0_0_0_3 * dm_kl[3];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            vj_ij[24] += R_0_0_1_0 * dm_kl[1];
            vj_ij[44] += R_0_0_1_0 * dm_kl[2];
            vj_ij[64] += R_0_0_1_0 * dm_kl[3];
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[25] += R_0_0_1_1 * dm_kl[1];
            vj_ij[45] += R_0_0_1_1 * dm_kl[2];
            vj_ij[65] += R_0_0_1_1 * dm_kl[3];
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[26] += R_0_0_1_2 * dm_kl[1];
            vj_ij[46] += R_0_0_1_2 * dm_kl[2];
            vj_ij[66] += R_0_0_1_2 * dm_kl[3];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            vj_ij[27] += R_0_0_2_0 * dm_kl[1];
            vj_ij[47] += R_0_0_2_0 * dm_kl[2];
            vj_ij[67] += R_0_0_2_0 * dm_kl[3];
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            vj_ij[28] += R_0_0_2_1 * dm_kl[1];
            vj_ij[48] += R_0_0_2_1 * dm_kl[2];
            vj_ij[68] += R_0_0_2_1 * dm_kl[3];
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[29] += R_0_0_3_0 * dm_kl[1];
            vj_ij[49] += R_0_0_3_0 * dm_kl[2];
            vj_ij[69] += R_0_0_3_0 * dm_kl[3];
            vj_ij[10] += R_0_1_0_0 * dm_kl[0];
            vj_ij[30] += R_0_1_0_0 * dm_kl[1];
            vj_ij[50] += R_0_1_0_0 * dm_kl[2];
            vj_ij[70] += R_0_1_0_0 * dm_kl[3];
            vj_ij[11] += R_0_1_0_1 * dm_kl[0];
            vj_ij[31] += R_0_1_0_1 * dm_kl[1];
            vj_ij[51] += R_0_1_0_1 * dm_kl[2];
            vj_ij[71] += R_0_1_0_1 * dm_kl[3];
            vj_ij[12] += R_0_1_0_2 * dm_kl[0];
            vj_ij[32] += R_0_1_0_2 * dm_kl[1];
            vj_ij[52] += R_0_1_0_2 * dm_kl[2];
            vj_ij[72] += R_0_1_0_2 * dm_kl[3];
            vj_ij[13] += R_0_1_1_0 * dm_kl[0];
            vj_ij[33] += R_0_1_1_0 * dm_kl[1];
            vj_ij[53] += R_0_1_1_0 * dm_kl[2];
            vj_ij[73] += R_0_1_1_0 * dm_kl[3];
            vj_ij[14] += R_0_1_1_1 * dm_kl[0];
            vj_ij[34] += R_0_1_1_1 * dm_kl[1];
            vj_ij[54] += R_0_1_1_1 * dm_kl[2];
            vj_ij[74] += R_0_1_1_1 * dm_kl[3];
            vj_ij[15] += R_0_1_2_0 * dm_kl[0];
            vj_ij[35] += R_0_1_2_0 * dm_kl[1];
            vj_ij[55] += R_0_1_2_0 * dm_kl[2];
            vj_ij[75] += R_0_1_2_0 * dm_kl[3];
            vj_ij[16] += R_0_2_0_0 * dm_kl[0];
            vj_ij[36] += R_0_2_0_0 * dm_kl[1];
            vj_ij[56] += R_0_2_0_0 * dm_kl[2];
            vj_ij[76] += R_0_2_0_0 * dm_kl[3];
            vj_ij[17] += R_0_2_0_1 * dm_kl[0];
            vj_ij[37] += R_0_2_0_1 * dm_kl[1];
            vj_ij[57] += R_0_2_0_1 * dm_kl[2];
            vj_ij[77] += R_0_2_0_1 * dm_kl[3];
            vj_ij[18] += R_0_2_1_0 * dm_kl[0];
            vj_ij[38] += R_0_2_1_0 * dm_kl[1];
            vj_ij[58] += R_0_2_1_0 * dm_kl[2];
            vj_ij[78] += R_0_2_1_0 * dm_kl[3];
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            vj_ij[39] += R_0_3_0_0 * dm_kl[1];
            vj_ij[59] += R_0_3_0_0 * dm_kl[2];
            vj_ij[79] += R_0_3_0_0 * dm_kl[3];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 20; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+20];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 20; ++n) {
                __syncthreads();
                for (int m = 0; m < 4; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+20*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 4; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 21; n += 16) {
        int kl = n / 21;
        int batch_kl = n - kl * 21;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 336 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*336+kl*336]);
            }
        }
    }
} }

// TILEX=48, TILEY=6
__global__ static
void md_j_4dm_3_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double vj_kl[4];
    double dm_kl[4];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1536;
    double *Rp_cache = Rq_cache + 384;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 1344 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 448; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 96; n += 256) {
        int task_kl = blockIdx.y * 96 + n;
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
            Rq_cache[n+96] = ykl;
            Rq_cache[n+192] = zkl;
            Rq_cache[n+288] = akl;
        } else {
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+96] = 1e5;
            Rq_cache[n+192] = 1e5;
            Rq_cache[n+288] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 1536; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 20 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 20;
            int i = n - i_dm * 20;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[80];
        for (int ij = 0; ij < 80; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 6; ++batch_kl) {
            int task_kl0 = blockIdx.y * 96 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*6] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
            __syncthreads();
            double xij = Rp_cache[tx+0];
            double yij = Rp_cache[tx+16];
            double zij = Rp_cache[tx+32];
            double aij = Rp_cache[tx+48];
            double xkl = Rq_cache[sq_kl+0];
            double ykl = Rq_cache[sq_kl+96];
            double zkl = Rq_cache[sq_kl+192];
            double akl = Rq_cache[sq_kl+288];
            fac = fac / (aij*akl*sqrt(aij+akl));
            double xpq = xij - xkl;
            double ypq = yij - ykl;
            double zpq = zij - zkl;
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * akl / (aij + akl);
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 4, 0, 256);
            if (remaining_n_dm == 1) {
            {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[96];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[128];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[144];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[240];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[288];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[48];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[80];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[96];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[112];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[128];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[144];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[160];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[176];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[192];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[208];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[224];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[240];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[256];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[272];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+96] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[64];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[96];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[112];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[128];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[144];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[160];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[176];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[192];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[208];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[224];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[256];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[272];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+192] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[64];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[80];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[96];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[112];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[128];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[144];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[160];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[176];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[192];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[208];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[224];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[256];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[272];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[288];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+288] += vj_kl[m];
            } }
            }{
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_ij[10] += R_0_1_0_0 * dm_kl[0];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[11] += R_0_1_0_1 * dm_kl[0];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[12] += R_0_1_0_2 * dm_kl[0];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[13] += R_0_1_1_0 * dm_kl[0];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[14] += R_0_1_1_1 * dm_kl[0];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[15] += R_0_1_2_0 * dm_kl[0];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_ij[16] += R_0_2_0_0 * dm_kl[0];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[17] += R_0_2_0_1 * dm_kl[0];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[18] += R_0_2_1_0 * dm_kl[0];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+1];
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[3] += R_0_0_0_4 * dm_kl[0];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[5] += R_0_0_1_2 * dm_kl[0];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[6] += R_0_0_1_3 * dm_kl[0];
            vj_ij[7] += R_0_0_2_1 * dm_kl[0];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[8] += R_0_0_2_2 * dm_kl[0];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[9] += R_0_0_3_1 * dm_kl[0];
            vj_ij[10] += R_0_1_0_1 * dm_kl[0];
            vj_ij[11] += R_0_1_0_2 * dm_kl[0];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[12] += R_0_1_0_3 * dm_kl[0];
            vj_ij[13] += R_0_1_1_1 * dm_kl[0];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[14] += R_0_1_1_2 * dm_kl[0];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[15] += R_0_1_2_1 * dm_kl[0];
            vj_ij[16] += R_0_2_0_1 * dm_kl[0];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[17] += R_0_2_0_2 * dm_kl[0];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[18] += R_0_2_1_1 * dm_kl[0];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[19] += R_0_3_0_1 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+2];
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[3] += R_0_0_1_3 * dm_kl[0];
            vj_ij[4] += R_0_0_2_0 * dm_kl[0];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[6] += R_0_0_2_2 * dm_kl[0];
            vj_ij[7] += R_0_0_3_0 * dm_kl[0];
            vj_ij[8] += R_0_0_3_1 * dm_kl[0];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[9] += R_0_0_4_0 * dm_kl[0];
            vj_ij[10] += R_0_1_1_0 * dm_kl[0];
            vj_ij[11] += R_0_1_1_1 * dm_kl[0];
            vj_ij[12] += R_0_1_1_2 * dm_kl[0];
            vj_ij[13] += R_0_1_2_0 * dm_kl[0];
            vj_ij[14] += R_0_1_2_1 * dm_kl[0];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[15] += R_0_1_3_0 * dm_kl[0];
            vj_ij[16] += R_0_2_1_0 * dm_kl[0];
            vj_ij[17] += R_0_2_1_1 * dm_kl[0];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[18] += R_0_2_2_0 * dm_kl[0];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[19] += R_0_3_1_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+3];
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[3] += R_0_1_0_3 * dm_kl[0];
            vj_ij[4] += R_0_1_1_0 * dm_kl[0];
            vj_ij[5] += R_0_1_1_1 * dm_kl[0];
            vj_ij[6] += R_0_1_1_2 * dm_kl[0];
            vj_ij[7] += R_0_1_2_0 * dm_kl[0];
            vj_ij[8] += R_0_1_2_1 * dm_kl[0];
            vj_ij[9] += R_0_1_3_0 * dm_kl[0];
            vj_ij[10] += R_0_2_0_0 * dm_kl[0];
            vj_ij[11] += R_0_2_0_1 * dm_kl[0];
            vj_ij[12] += R_0_2_0_2 * dm_kl[0];
            vj_ij[13] += R_0_2_1_0 * dm_kl[0];
            vj_ij[14] += R_0_2_1_1 * dm_kl[0];
            vj_ij[15] += R_0_2_2_0 * dm_kl[0];
            vj_ij[16] += R_0_3_0_0 * dm_kl[0];
            vj_ij[17] += R_0_3_0_1 * dm_kl[0];
            vj_ij[18] += R_0_3_1_0 * dm_kl[0];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[19] += R_0_4_0_0 * dm_kl[0];
            }
            } else if (remaining_n_dm == 2) {
            {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[320];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[336];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[352];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[368];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[384];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[400];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[416];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[432];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[448];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[464];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[480];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[496];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[512];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[528];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[544];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[560];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[576];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[592];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[608];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[320];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[336];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[352];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_0_4 * dm_ij_cache[368];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[384];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[400];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[96];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[416];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[432];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[128];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[448];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[144];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[464];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[160];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[480];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[176];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[496];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[192];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[512];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[208];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[528];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[224];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[544];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[240];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[560];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[256];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[576];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[272];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[592];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[288];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[608];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[304];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+96] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[320];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[336];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[352];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[368];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[384];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[400];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[96];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[416];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[112];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[432];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[448];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_0_4_0 * dm_ij_cache[464];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[160];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[176];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[496];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[192];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[512];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[208];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[528];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[224];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[544];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[240];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[256];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[576];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[272];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[592];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[288];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[608];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[304];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+192] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[320];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[336];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[352];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[368];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[64];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[384];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[400];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[416];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[432];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[448];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[464];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[160];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[176];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[496];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[192];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[512];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[208];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[528];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[224];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[544];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[240];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[256];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[576];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[272];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[592];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[288];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[608];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[304];
            vj_kl[1] -= R_0_4_0_0 * dm_ij_cache[624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+288] += vj_kl[m];
            } }
            }{
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[1];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[21] += R_0_0_0_1 * dm_kl[1];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[22] += R_0_0_0_2 * dm_kl[1];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[23] += R_0_0_0_3 * dm_kl[1];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            vj_ij[24] += R_0_0_1_0 * dm_kl[1];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[25] += R_0_0_1_1 * dm_kl[1];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[26] += R_0_0_1_2 * dm_kl[1];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            vj_ij[27] += R_0_0_2_0 * dm_kl[1];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            vj_ij[28] += R_0_0_2_1 * dm_kl[1];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[29] += R_0_0_3_0 * dm_kl[1];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_ij[10] += R_0_1_0_0 * dm_kl[0];
            vj_ij[30] += R_0_1_0_0 * dm_kl[1];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[11] += R_0_1_0_1 * dm_kl[0];
            vj_ij[31] += R_0_1_0_1 * dm_kl[1];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[12] += R_0_1_0_2 * dm_kl[0];
            vj_ij[32] += R_0_1_0_2 * dm_kl[1];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[13] += R_0_1_1_0 * dm_kl[0];
            vj_ij[33] += R_0_1_1_0 * dm_kl[1];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[14] += R_0_1_1_1 * dm_kl[0];
            vj_ij[34] += R_0_1_1_1 * dm_kl[1];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[15] += R_0_1_2_0 * dm_kl[0];
            vj_ij[35] += R_0_1_2_0 * dm_kl[1];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_ij[16] += R_0_2_0_0 * dm_kl[0];
            vj_ij[36] += R_0_2_0_0 * dm_kl[1];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[17] += R_0_2_0_1 * dm_kl[0];
            vj_ij[37] += R_0_2_0_1 * dm_kl[1];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[18] += R_0_2_1_0 * dm_kl[0];
            vj_ij[38] += R_0_2_1_0 * dm_kl[1];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            vj_ij[39] += R_0_3_0_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[20] += R_0_0_0_1 * dm_kl[1];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[21] += R_0_0_0_2 * dm_kl[1];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[22] += R_0_0_0_3 * dm_kl[1];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[3] += R_0_0_0_4 * dm_kl[0];
            vj_ij[23] += R_0_0_0_4 * dm_kl[1];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[24] += R_0_0_1_1 * dm_kl[1];
            vj_ij[5] += R_0_0_1_2 * dm_kl[0];
            vj_ij[25] += R_0_0_1_2 * dm_kl[1];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[6] += R_0_0_1_3 * dm_kl[0];
            vj_ij[26] += R_0_0_1_3 * dm_kl[1];
            vj_ij[7] += R_0_0_2_1 * dm_kl[0];
            vj_ij[27] += R_0_0_2_1 * dm_kl[1];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[8] += R_0_0_2_2 * dm_kl[0];
            vj_ij[28] += R_0_0_2_2 * dm_kl[1];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[9] += R_0_0_3_1 * dm_kl[0];
            vj_ij[29] += R_0_0_3_1 * dm_kl[1];
            vj_ij[10] += R_0_1_0_1 * dm_kl[0];
            vj_ij[30] += R_0_1_0_1 * dm_kl[1];
            vj_ij[11] += R_0_1_0_2 * dm_kl[0];
            vj_ij[31] += R_0_1_0_2 * dm_kl[1];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[12] += R_0_1_0_3 * dm_kl[0];
            vj_ij[32] += R_0_1_0_3 * dm_kl[1];
            vj_ij[13] += R_0_1_1_1 * dm_kl[0];
            vj_ij[33] += R_0_1_1_1 * dm_kl[1];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[14] += R_0_1_1_2 * dm_kl[0];
            vj_ij[34] += R_0_1_1_2 * dm_kl[1];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[15] += R_0_1_2_1 * dm_kl[0];
            vj_ij[35] += R_0_1_2_1 * dm_kl[1];
            vj_ij[16] += R_0_2_0_1 * dm_kl[0];
            vj_ij[36] += R_0_2_0_1 * dm_kl[1];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[17] += R_0_2_0_2 * dm_kl[0];
            vj_ij[37] += R_0_2_0_2 * dm_kl[1];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[18] += R_0_2_1_1 * dm_kl[0];
            vj_ij[38] += R_0_2_1_1 * dm_kl[1];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[19] += R_0_3_0_1 * dm_kl[0];
            vj_ij[39] += R_0_3_0_1 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2]; }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[20] += R_0_0_1_0 * dm_kl[1];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[21] += R_0_0_1_1 * dm_kl[1];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[22] += R_0_0_1_2 * dm_kl[1];
            vj_ij[3] += R_0_0_1_3 * dm_kl[0];
            vj_ij[23] += R_0_0_1_3 * dm_kl[1];
            vj_ij[4] += R_0_0_2_0 * dm_kl[0];
            vj_ij[24] += R_0_0_2_0 * dm_kl[1];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[25] += R_0_0_2_1 * dm_kl[1];
            vj_ij[6] += R_0_0_2_2 * dm_kl[0];
            vj_ij[26] += R_0_0_2_2 * dm_kl[1];
            vj_ij[7] += R_0_0_3_0 * dm_kl[0];
            vj_ij[27] += R_0_0_3_0 * dm_kl[1];
            vj_ij[8] += R_0_0_3_1 * dm_kl[0];
            vj_ij[28] += R_0_0_3_1 * dm_kl[1];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[9] += R_0_0_4_0 * dm_kl[0];
            vj_ij[29] += R_0_0_4_0 * dm_kl[1];
            vj_ij[10] += R_0_1_1_0 * dm_kl[0];
            vj_ij[30] += R_0_1_1_0 * dm_kl[1];
            vj_ij[11] += R_0_1_1_1 * dm_kl[0];
            vj_ij[31] += R_0_1_1_1 * dm_kl[1];
            vj_ij[12] += R_0_1_1_2 * dm_kl[0];
            vj_ij[32] += R_0_1_1_2 * dm_kl[1];
            vj_ij[13] += R_0_1_2_0 * dm_kl[0];
            vj_ij[33] += R_0_1_2_0 * dm_kl[1];
            vj_ij[14] += R_0_1_2_1 * dm_kl[0];
            vj_ij[34] += R_0_1_2_1 * dm_kl[1];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[15] += R_0_1_3_0 * dm_kl[0];
            vj_ij[35] += R_0_1_3_0 * dm_kl[1];
            vj_ij[16] += R_0_2_1_0 * dm_kl[0];
            vj_ij[36] += R_0_2_1_0 * dm_kl[1];
            vj_ij[17] += R_0_2_1_1 * dm_kl[0];
            vj_ij[37] += R_0_2_1_1 * dm_kl[1];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[18] += R_0_2_2_0 * dm_kl[0];
            vj_ij[38] += R_0_2_2_0 * dm_kl[1];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[19] += R_0_3_1_0 * dm_kl[0];
            vj_ij[39] += R_0_3_1_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[20] += R_0_1_0_0 * dm_kl[1];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[21] += R_0_1_0_1 * dm_kl[1];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[22] += R_0_1_0_2 * dm_kl[1];
            vj_ij[3] += R_0_1_0_3 * dm_kl[0];
            vj_ij[23] += R_0_1_0_3 * dm_kl[1];
            vj_ij[4] += R_0_1_1_0 * dm_kl[0];
            vj_ij[24] += R_0_1_1_0 * dm_kl[1];
            vj_ij[5] += R_0_1_1_1 * dm_kl[0];
            vj_ij[25] += R_0_1_1_1 * dm_kl[1];
            vj_ij[6] += R_0_1_1_2 * dm_kl[0];
            vj_ij[26] += R_0_1_1_2 * dm_kl[1];
            vj_ij[7] += R_0_1_2_0 * dm_kl[0];
            vj_ij[27] += R_0_1_2_0 * dm_kl[1];
            vj_ij[8] += R_0_1_2_1 * dm_kl[0];
            vj_ij[28] += R_0_1_2_1 * dm_kl[1];
            vj_ij[9] += R_0_1_3_0 * dm_kl[0];
            vj_ij[29] += R_0_1_3_0 * dm_kl[1];
            vj_ij[10] += R_0_2_0_0 * dm_kl[0];
            vj_ij[30] += R_0_2_0_0 * dm_kl[1];
            vj_ij[11] += R_0_2_0_1 * dm_kl[0];
            vj_ij[31] += R_0_2_0_1 * dm_kl[1];
            vj_ij[12] += R_0_2_0_2 * dm_kl[0];
            vj_ij[32] += R_0_2_0_2 * dm_kl[1];
            vj_ij[13] += R_0_2_1_0 * dm_kl[0];
            vj_ij[33] += R_0_2_1_0 * dm_kl[1];
            vj_ij[14] += R_0_2_1_1 * dm_kl[0];
            vj_ij[34] += R_0_2_1_1 * dm_kl[1];
            vj_ij[15] += R_0_2_2_0 * dm_kl[0];
            vj_ij[35] += R_0_2_2_0 * dm_kl[1];
            vj_ij[16] += R_0_3_0_0 * dm_kl[0];
            vj_ij[36] += R_0_3_0_0 * dm_kl[1];
            vj_ij[17] += R_0_3_0_1 * dm_kl[0];
            vj_ij[37] += R_0_3_0_1 * dm_kl[1];
            vj_ij[18] += R_0_3_1_0 * dm_kl[0];
            vj_ij[38] += R_0_3_1_0 * dm_kl[1];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[19] += R_0_4_0_0 * dm_kl[0];
            vj_ij[39] += R_0_4_0_0 * dm_kl[1];
            }
            } else {
            {
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[320];
            vj_kl[2] += gamma_inc[0*256] * dm_ij_cache[640];
            vj_kl[3] += gamma_inc[0*256] * dm_ij_cache[960];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[336];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[656];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[976];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[352];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[672];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[992];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[368];
            vj_kl[2] += R_0_0_0_3 * dm_ij_cache[688];
            vj_kl[3] += R_0_0_0_3 * dm_ij_cache[1008];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[384];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[704];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[1024];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[400];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[720];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[1040];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[416];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[736];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[1056];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[432];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[752];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[1072];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[448];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[768];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[1088];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[464];
            vj_kl[2] += R_0_0_3_0 * dm_ij_cache[784];
            vj_kl[3] += R_0_0_3_0 * dm_ij_cache[1104];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[480];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[800];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[1120];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[496];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[816];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[1136];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[512];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[832];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[1152];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[528];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[848];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[1168];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[544];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[864];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[1184];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[560];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[880];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[1200];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[576];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[896];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[1216];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[592];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[912];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[1232];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[608];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[928];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[1248];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[624];
            vj_kl[2] += R_0_3_0_0 * dm_ij_cache[944];
            vj_kl[3] += R_0_3_0_0 * dm_ij_cache[1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[320];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[640];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[960];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[336];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[656];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[976];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[352];
            vj_kl[2] -= R_0_0_0_3 * dm_ij_cache[672];
            vj_kl[3] -= R_0_0_0_3 * dm_ij_cache[992];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_0_4 * dm_ij_cache[368];
            vj_kl[2] -= R_0_0_0_4 * dm_ij_cache[688];
            vj_kl[3] -= R_0_0_0_4 * dm_ij_cache[1008];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[384];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[704];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[1024];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[400];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[720];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[1040];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[96];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[416];
            vj_kl[2] -= R_0_0_1_3 * dm_ij_cache[736];
            vj_kl[3] -= R_0_0_1_3 * dm_ij_cache[1056];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[112];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[432];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[752];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[1072];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[128];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[448];
            vj_kl[2] -= R_0_0_2_2 * dm_ij_cache[768];
            vj_kl[3] -= R_0_0_2_2 * dm_ij_cache[1088];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[144];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[464];
            vj_kl[2] -= R_0_0_3_1 * dm_ij_cache[784];
            vj_kl[3] -= R_0_0_3_1 * dm_ij_cache[1104];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[160];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[480];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[800];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[1120];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[176];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[496];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[816];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[1136];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[192];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[512];
            vj_kl[2] -= R_0_1_0_3 * dm_ij_cache[832];
            vj_kl[3] -= R_0_1_0_3 * dm_ij_cache[1152];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[208];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[528];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[848];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[1168];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[224];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[544];
            vj_kl[2] -= R_0_1_1_2 * dm_ij_cache[864];
            vj_kl[3] -= R_0_1_1_2 * dm_ij_cache[1184];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[240];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[560];
            vj_kl[2] -= R_0_1_2_1 * dm_ij_cache[880];
            vj_kl[3] -= R_0_1_2_1 * dm_ij_cache[1200];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[256];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[576];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[896];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[1216];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[272];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[592];
            vj_kl[2] -= R_0_2_0_2 * dm_ij_cache[912];
            vj_kl[3] -= R_0_2_0_2 * dm_ij_cache[1232];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[288];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[608];
            vj_kl[2] -= R_0_2_1_1 * dm_ij_cache[928];
            vj_kl[3] -= R_0_2_1_1 * dm_ij_cache[1248];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[304];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[624];
            vj_kl[2] -= R_0_3_0_1 * dm_ij_cache[944];
            vj_kl[3] -= R_0_3_0_1 * dm_ij_cache[1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+96] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[320];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[640];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[960];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[336];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[656];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[976];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[352];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[672];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[992];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[368];
            vj_kl[2] -= R_0_0_1_3 * dm_ij_cache[688];
            vj_kl[3] -= R_0_0_1_3 * dm_ij_cache[1008];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[384];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[704];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[1024];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[400];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[720];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[1040];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[96];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[416];
            vj_kl[2] -= R_0_0_2_2 * dm_ij_cache[736];
            vj_kl[3] -= R_0_0_2_2 * dm_ij_cache[1056];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[112];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[432];
            vj_kl[2] -= R_0_0_3_0 * dm_ij_cache[752];
            vj_kl[3] -= R_0_0_3_0 * dm_ij_cache[1072];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[448];
            vj_kl[2] -= R_0_0_3_1 * dm_ij_cache[768];
            vj_kl[3] -= R_0_0_3_1 * dm_ij_cache[1088];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_0_4_0 * dm_ij_cache[464];
            vj_kl[2] -= R_0_0_4_0 * dm_ij_cache[784];
            vj_kl[3] -= R_0_0_4_0 * dm_ij_cache[1104];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[160];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[480];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[800];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[1120];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[176];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[496];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[816];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[1136];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[192];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[512];
            vj_kl[2] -= R_0_1_1_2 * dm_ij_cache[832];
            vj_kl[3] -= R_0_1_1_2 * dm_ij_cache[1152];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[208];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[528];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[848];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[1168];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[224];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[544];
            vj_kl[2] -= R_0_1_2_1 * dm_ij_cache[864];
            vj_kl[3] -= R_0_1_2_1 * dm_ij_cache[1184];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[240];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[560];
            vj_kl[2] -= R_0_1_3_0 * dm_ij_cache[880];
            vj_kl[3] -= R_0_1_3_0 * dm_ij_cache[1200];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[256];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[576];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[896];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[1216];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[272];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[592];
            vj_kl[2] -= R_0_2_1_1 * dm_ij_cache[912];
            vj_kl[3] -= R_0_2_1_1 * dm_ij_cache[1232];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[288];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[608];
            vj_kl[2] -= R_0_2_2_0 * dm_ij_cache[928];
            vj_kl[3] -= R_0_2_2_0 * dm_ij_cache[1248];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[304];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[624];
            vj_kl[2] -= R_0_3_1_0 * dm_ij_cache[944];
            vj_kl[3] -= R_0_3_1_0 * dm_ij_cache[1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+192] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[320];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[640];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[960];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[336];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[656];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[976];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[352];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[672];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[992];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[368];
            vj_kl[2] -= R_0_1_0_3 * dm_ij_cache[688];
            vj_kl[3] -= R_0_1_0_3 * dm_ij_cache[1008];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[64];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[384];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[704];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[1024];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[400];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[720];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[1040];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[416];
            vj_kl[2] -= R_0_1_1_2 * dm_ij_cache[736];
            vj_kl[3] -= R_0_1_1_2 * dm_ij_cache[1056];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[432];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[752];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[1072];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[448];
            vj_kl[2] -= R_0_1_2_1 * dm_ij_cache[768];
            vj_kl[3] -= R_0_1_2_1 * dm_ij_cache[1088];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[464];
            vj_kl[2] -= R_0_1_3_0 * dm_ij_cache[784];
            vj_kl[3] -= R_0_1_3_0 * dm_ij_cache[1104];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[160];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[480];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[800];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[1120];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[176];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[496];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[816];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[1136];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[192];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[512];
            vj_kl[2] -= R_0_2_0_2 * dm_ij_cache[832];
            vj_kl[3] -= R_0_2_0_2 * dm_ij_cache[1152];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[208];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[528];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[848];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[1168];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[224];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[544];
            vj_kl[2] -= R_0_2_1_1 * dm_ij_cache[864];
            vj_kl[3] -= R_0_2_1_1 * dm_ij_cache[1184];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[240];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[560];
            vj_kl[2] -= R_0_2_2_0 * dm_ij_cache[880];
            vj_kl[3] -= R_0_2_2_0 * dm_ij_cache[1200];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[256];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[576];
            vj_kl[2] -= R_0_3_0_0 * dm_ij_cache[896];
            vj_kl[3] -= R_0_3_0_0 * dm_ij_cache[1216];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[272];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[592];
            vj_kl[2] -= R_0_3_0_1 * dm_ij_cache[912];
            vj_kl[3] -= R_0_3_0_1 * dm_ij_cache[1232];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[288];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[608];
            vj_kl[2] -= R_0_3_1_0 * dm_ij_cache[928];
            vj_kl[3] -= R_0_3_1_0 * dm_ij_cache[1248];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[304];
            vj_kl[1] -= R_0_4_0_0 * dm_ij_cache[624];
            vj_kl[2] -= R_0_4_0_0 * dm_ij_cache[944];
            vj_kl[3] -= R_0_4_0_0 * dm_ij_cache[1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*384+288] += vj_kl[m];
            } }
            }{
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[40] += gamma_inc[0*256] * dm_kl[2];
            vj_ij[60] += gamma_inc[0*256] * dm_kl[3];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[21] += R_0_0_0_1 * dm_kl[1];
            vj_ij[41] += R_0_0_0_1 * dm_kl[2];
            vj_ij[61] += R_0_0_0_1 * dm_kl[3];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[22] += R_0_0_0_2 * dm_kl[1];
            vj_ij[42] += R_0_0_0_2 * dm_kl[2];
            vj_ij[62] += R_0_0_0_2 * dm_kl[3];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[23] += R_0_0_0_3 * dm_kl[1];
            vj_ij[43] += R_0_0_0_3 * dm_kl[2];
            vj_ij[63] += R_0_0_0_3 * dm_kl[3];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            vj_ij[24] += R_0_0_1_0 * dm_kl[1];
            vj_ij[44] += R_0_0_1_0 * dm_kl[2];
            vj_ij[64] += R_0_0_1_0 * dm_kl[3];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[25] += R_0_0_1_1 * dm_kl[1];
            vj_ij[45] += R_0_0_1_1 * dm_kl[2];
            vj_ij[65] += R_0_0_1_1 * dm_kl[3];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[26] += R_0_0_1_2 * dm_kl[1];
            vj_ij[46] += R_0_0_1_2 * dm_kl[2];
            vj_ij[66] += R_0_0_1_2 * dm_kl[3];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            vj_ij[27] += R_0_0_2_0 * dm_kl[1];
            vj_ij[47] += R_0_0_2_0 * dm_kl[2];
            vj_ij[67] += R_0_0_2_0 * dm_kl[3];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            vj_ij[28] += R_0_0_2_1 * dm_kl[1];
            vj_ij[48] += R_0_0_2_1 * dm_kl[2];
            vj_ij[68] += R_0_0_2_1 * dm_kl[3];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[29] += R_0_0_3_0 * dm_kl[1];
            vj_ij[49] += R_0_0_3_0 * dm_kl[2];
            vj_ij[69] += R_0_0_3_0 * dm_kl[3];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_ij[10] += R_0_1_0_0 * dm_kl[0];
            vj_ij[30] += R_0_1_0_0 * dm_kl[1];
            vj_ij[50] += R_0_1_0_0 * dm_kl[2];
            vj_ij[70] += R_0_1_0_0 * dm_kl[3];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[11] += R_0_1_0_1 * dm_kl[0];
            vj_ij[31] += R_0_1_0_1 * dm_kl[1];
            vj_ij[51] += R_0_1_0_1 * dm_kl[2];
            vj_ij[71] += R_0_1_0_1 * dm_kl[3];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[12] += R_0_1_0_2 * dm_kl[0];
            vj_ij[32] += R_0_1_0_2 * dm_kl[1];
            vj_ij[52] += R_0_1_0_2 * dm_kl[2];
            vj_ij[72] += R_0_1_0_2 * dm_kl[3];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[13] += R_0_1_1_0 * dm_kl[0];
            vj_ij[33] += R_0_1_1_0 * dm_kl[1];
            vj_ij[53] += R_0_1_1_0 * dm_kl[2];
            vj_ij[73] += R_0_1_1_0 * dm_kl[3];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[14] += R_0_1_1_1 * dm_kl[0];
            vj_ij[34] += R_0_1_1_1 * dm_kl[1];
            vj_ij[54] += R_0_1_1_1 * dm_kl[2];
            vj_ij[74] += R_0_1_1_1 * dm_kl[3];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[15] += R_0_1_2_0 * dm_kl[0];
            vj_ij[35] += R_0_1_2_0 * dm_kl[1];
            vj_ij[55] += R_0_1_2_0 * dm_kl[2];
            vj_ij[75] += R_0_1_2_0 * dm_kl[3];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_ij[16] += R_0_2_0_0 * dm_kl[0];
            vj_ij[36] += R_0_2_0_0 * dm_kl[1];
            vj_ij[56] += R_0_2_0_0 * dm_kl[2];
            vj_ij[76] += R_0_2_0_0 * dm_kl[3];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[17] += R_0_2_0_1 * dm_kl[0];
            vj_ij[37] += R_0_2_0_1 * dm_kl[1];
            vj_ij[57] += R_0_2_0_1 * dm_kl[2];
            vj_ij[77] += R_0_2_0_1 * dm_kl[3];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[18] += R_0_2_1_0 * dm_kl[0];
            vj_ij[38] += R_0_2_1_0 * dm_kl[1];
            vj_ij[58] += R_0_2_1_0 * dm_kl[2];
            vj_ij[78] += R_0_2_1_0 * dm_kl[3];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            vj_ij[39] += R_0_3_0_0 * dm_kl[1];
            vj_ij[59] += R_0_3_0_0 * dm_kl[2];
            vj_ij[79] += R_0_3_0_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1];
            }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[20] += R_0_0_0_1 * dm_kl[1];
            vj_ij[40] += R_0_0_0_1 * dm_kl[2];
            vj_ij[60] += R_0_0_0_1 * dm_kl[3];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[21] += R_0_0_0_2 * dm_kl[1];
            vj_ij[41] += R_0_0_0_2 * dm_kl[2];
            vj_ij[61] += R_0_0_0_2 * dm_kl[3];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[22] += R_0_0_0_3 * dm_kl[1];
            vj_ij[42] += R_0_0_0_3 * dm_kl[2];
            vj_ij[62] += R_0_0_0_3 * dm_kl[3];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[3] += R_0_0_0_4 * dm_kl[0];
            vj_ij[23] += R_0_0_0_4 * dm_kl[1];
            vj_ij[43] += R_0_0_0_4 * dm_kl[2];
            vj_ij[63] += R_0_0_0_4 * dm_kl[3];
            vj_ij[4] += R_0_0_1_1 * dm_kl[0];
            vj_ij[24] += R_0_0_1_1 * dm_kl[1];
            vj_ij[44] += R_0_0_1_1 * dm_kl[2];
            vj_ij[64] += R_0_0_1_1 * dm_kl[3];
            vj_ij[5] += R_0_0_1_2 * dm_kl[0];
            vj_ij[25] += R_0_0_1_2 * dm_kl[1];
            vj_ij[45] += R_0_0_1_2 * dm_kl[2];
            vj_ij[65] += R_0_0_1_2 * dm_kl[3];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[6] += R_0_0_1_3 * dm_kl[0];
            vj_ij[26] += R_0_0_1_3 * dm_kl[1];
            vj_ij[46] += R_0_0_1_3 * dm_kl[2];
            vj_ij[66] += R_0_0_1_3 * dm_kl[3];
            vj_ij[7] += R_0_0_2_1 * dm_kl[0];
            vj_ij[27] += R_0_0_2_1 * dm_kl[1];
            vj_ij[47] += R_0_0_2_1 * dm_kl[2];
            vj_ij[67] += R_0_0_2_1 * dm_kl[3];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[8] += R_0_0_2_2 * dm_kl[0];
            vj_ij[28] += R_0_0_2_2 * dm_kl[1];
            vj_ij[48] += R_0_0_2_2 * dm_kl[2];
            vj_ij[68] += R_0_0_2_2 * dm_kl[3];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[9] += R_0_0_3_1 * dm_kl[0];
            vj_ij[29] += R_0_0_3_1 * dm_kl[1];
            vj_ij[49] += R_0_0_3_1 * dm_kl[2];
            vj_ij[69] += R_0_0_3_1 * dm_kl[3];
            vj_ij[10] += R_0_1_0_1 * dm_kl[0];
            vj_ij[30] += R_0_1_0_1 * dm_kl[1];
            vj_ij[50] += R_0_1_0_1 * dm_kl[2];
            vj_ij[70] += R_0_1_0_1 * dm_kl[3];
            vj_ij[11] += R_0_1_0_2 * dm_kl[0];
            vj_ij[31] += R_0_1_0_2 * dm_kl[1];
            vj_ij[51] += R_0_1_0_2 * dm_kl[2];
            vj_ij[71] += R_0_1_0_2 * dm_kl[3];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[12] += R_0_1_0_3 * dm_kl[0];
            vj_ij[32] += R_0_1_0_3 * dm_kl[1];
            vj_ij[52] += R_0_1_0_3 * dm_kl[2];
            vj_ij[72] += R_0_1_0_3 * dm_kl[3];
            vj_ij[13] += R_0_1_1_1 * dm_kl[0];
            vj_ij[33] += R_0_1_1_1 * dm_kl[1];
            vj_ij[53] += R_0_1_1_1 * dm_kl[2];
            vj_ij[73] += R_0_1_1_1 * dm_kl[3];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[14] += R_0_1_1_2 * dm_kl[0];
            vj_ij[34] += R_0_1_1_2 * dm_kl[1];
            vj_ij[54] += R_0_1_1_2 * dm_kl[2];
            vj_ij[74] += R_0_1_1_2 * dm_kl[3];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[15] += R_0_1_2_1 * dm_kl[0];
            vj_ij[35] += R_0_1_2_1 * dm_kl[1];
            vj_ij[55] += R_0_1_2_1 * dm_kl[2];
            vj_ij[75] += R_0_1_2_1 * dm_kl[3];
            vj_ij[16] += R_0_2_0_1 * dm_kl[0];
            vj_ij[36] += R_0_2_0_1 * dm_kl[1];
            vj_ij[56] += R_0_2_0_1 * dm_kl[2];
            vj_ij[76] += R_0_2_0_1 * dm_kl[3];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[17] += R_0_2_0_2 * dm_kl[0];
            vj_ij[37] += R_0_2_0_2 * dm_kl[1];
            vj_ij[57] += R_0_2_0_2 * dm_kl[2];
            vj_ij[77] += R_0_2_0_2 * dm_kl[3];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[18] += R_0_2_1_1 * dm_kl[0];
            vj_ij[38] += R_0_2_1_1 * dm_kl[1];
            vj_ij[58] += R_0_2_1_1 * dm_kl[2];
            vj_ij[78] += R_0_2_1_1 * dm_kl[3];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[19] += R_0_3_0_1 * dm_kl[0];
            vj_ij[39] += R_0_3_0_1 * dm_kl[1];
            vj_ij[59] += R_0_3_0_1 * dm_kl[2];
            vj_ij[79] += R_0_3_0_1 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2];
            }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[20] += R_0_0_1_0 * dm_kl[1];
            vj_ij[40] += R_0_0_1_0 * dm_kl[2];
            vj_ij[60] += R_0_0_1_0 * dm_kl[3];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[21] += R_0_0_1_1 * dm_kl[1];
            vj_ij[41] += R_0_0_1_1 * dm_kl[2];
            vj_ij[61] += R_0_0_1_1 * dm_kl[3];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[22] += R_0_0_1_2 * dm_kl[1];
            vj_ij[42] += R_0_0_1_2 * dm_kl[2];
            vj_ij[62] += R_0_0_1_2 * dm_kl[3];
            vj_ij[3] += R_0_0_1_3 * dm_kl[0];
            vj_ij[23] += R_0_0_1_3 * dm_kl[1];
            vj_ij[43] += R_0_0_1_3 * dm_kl[2];
            vj_ij[63] += R_0_0_1_3 * dm_kl[3];
            vj_ij[4] += R_0_0_2_0 * dm_kl[0];
            vj_ij[24] += R_0_0_2_0 * dm_kl[1];
            vj_ij[44] += R_0_0_2_0 * dm_kl[2];
            vj_ij[64] += R_0_0_2_0 * dm_kl[3];
            vj_ij[5] += R_0_0_2_1 * dm_kl[0];
            vj_ij[25] += R_0_0_2_1 * dm_kl[1];
            vj_ij[45] += R_0_0_2_1 * dm_kl[2];
            vj_ij[65] += R_0_0_2_1 * dm_kl[3];
            vj_ij[6] += R_0_0_2_2 * dm_kl[0];
            vj_ij[26] += R_0_0_2_2 * dm_kl[1];
            vj_ij[46] += R_0_0_2_2 * dm_kl[2];
            vj_ij[66] += R_0_0_2_2 * dm_kl[3];
            vj_ij[7] += R_0_0_3_0 * dm_kl[0];
            vj_ij[27] += R_0_0_3_0 * dm_kl[1];
            vj_ij[47] += R_0_0_3_0 * dm_kl[2];
            vj_ij[67] += R_0_0_3_0 * dm_kl[3];
            vj_ij[8] += R_0_0_3_1 * dm_kl[0];
            vj_ij[28] += R_0_0_3_1 * dm_kl[1];
            vj_ij[48] += R_0_0_3_1 * dm_kl[2];
            vj_ij[68] += R_0_0_3_1 * dm_kl[3];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[9] += R_0_0_4_0 * dm_kl[0];
            vj_ij[29] += R_0_0_4_0 * dm_kl[1];
            vj_ij[49] += R_0_0_4_0 * dm_kl[2];
            vj_ij[69] += R_0_0_4_0 * dm_kl[3];
            vj_ij[10] += R_0_1_1_0 * dm_kl[0];
            vj_ij[30] += R_0_1_1_0 * dm_kl[1];
            vj_ij[50] += R_0_1_1_0 * dm_kl[2];
            vj_ij[70] += R_0_1_1_0 * dm_kl[3];
            vj_ij[11] += R_0_1_1_1 * dm_kl[0];
            vj_ij[31] += R_0_1_1_1 * dm_kl[1];
            vj_ij[51] += R_0_1_1_1 * dm_kl[2];
            vj_ij[71] += R_0_1_1_1 * dm_kl[3];
            vj_ij[12] += R_0_1_1_2 * dm_kl[0];
            vj_ij[32] += R_0_1_1_2 * dm_kl[1];
            vj_ij[52] += R_0_1_1_2 * dm_kl[2];
            vj_ij[72] += R_0_1_1_2 * dm_kl[3];
            vj_ij[13] += R_0_1_2_0 * dm_kl[0];
            vj_ij[33] += R_0_1_2_0 * dm_kl[1];
            vj_ij[53] += R_0_1_2_0 * dm_kl[2];
            vj_ij[73] += R_0_1_2_0 * dm_kl[3];
            vj_ij[14] += R_0_1_2_1 * dm_kl[0];
            vj_ij[34] += R_0_1_2_1 * dm_kl[1];
            vj_ij[54] += R_0_1_2_1 * dm_kl[2];
            vj_ij[74] += R_0_1_2_1 * dm_kl[3];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[15] += R_0_1_3_0 * dm_kl[0];
            vj_ij[35] += R_0_1_3_0 * dm_kl[1];
            vj_ij[55] += R_0_1_3_0 * dm_kl[2];
            vj_ij[75] += R_0_1_3_0 * dm_kl[3];
            vj_ij[16] += R_0_2_1_0 * dm_kl[0];
            vj_ij[36] += R_0_2_1_0 * dm_kl[1];
            vj_ij[56] += R_0_2_1_0 * dm_kl[2];
            vj_ij[76] += R_0_2_1_0 * dm_kl[3];
            vj_ij[17] += R_0_2_1_1 * dm_kl[0];
            vj_ij[37] += R_0_2_1_1 * dm_kl[1];
            vj_ij[57] += R_0_2_1_1 * dm_kl[2];
            vj_ij[77] += R_0_2_1_1 * dm_kl[3];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[18] += R_0_2_2_0 * dm_kl[0];
            vj_ij[38] += R_0_2_2_0 * dm_kl[1];
            vj_ij[58] += R_0_2_2_0 * dm_kl[2];
            vj_ij[78] += R_0_2_2_0 * dm_kl[3];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[19] += R_0_3_1_0 * dm_kl[0];
            vj_ij[39] += R_0_3_1_0 * dm_kl[1];
            vj_ij[59] += R_0_3_1_0 * dm_kl[2];
            vj_ij[79] += R_0_3_1_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3];
            }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[20] += R_0_1_0_0 * dm_kl[1];
            vj_ij[40] += R_0_1_0_0 * dm_kl[2];
            vj_ij[60] += R_0_1_0_0 * dm_kl[3];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[21] += R_0_1_0_1 * dm_kl[1];
            vj_ij[41] += R_0_1_0_1 * dm_kl[2];
            vj_ij[61] += R_0_1_0_1 * dm_kl[3];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[22] += R_0_1_0_2 * dm_kl[1];
            vj_ij[42] += R_0_1_0_2 * dm_kl[2];
            vj_ij[62] += R_0_1_0_2 * dm_kl[3];
            vj_ij[3] += R_0_1_0_3 * dm_kl[0];
            vj_ij[23] += R_0_1_0_3 * dm_kl[1];
            vj_ij[43] += R_0_1_0_3 * dm_kl[2];
            vj_ij[63] += R_0_1_0_3 * dm_kl[3];
            vj_ij[4] += R_0_1_1_0 * dm_kl[0];
            vj_ij[24] += R_0_1_1_0 * dm_kl[1];
            vj_ij[44] += R_0_1_1_0 * dm_kl[2];
            vj_ij[64] += R_0_1_1_0 * dm_kl[3];
            vj_ij[5] += R_0_1_1_1 * dm_kl[0];
            vj_ij[25] += R_0_1_1_1 * dm_kl[1];
            vj_ij[45] += R_0_1_1_1 * dm_kl[2];
            vj_ij[65] += R_0_1_1_1 * dm_kl[3];
            vj_ij[6] += R_0_1_1_2 * dm_kl[0];
            vj_ij[26] += R_0_1_1_2 * dm_kl[1];
            vj_ij[46] += R_0_1_1_2 * dm_kl[2];
            vj_ij[66] += R_0_1_1_2 * dm_kl[3];
            vj_ij[7] += R_0_1_2_0 * dm_kl[0];
            vj_ij[27] += R_0_1_2_0 * dm_kl[1];
            vj_ij[47] += R_0_1_2_0 * dm_kl[2];
            vj_ij[67] += R_0_1_2_0 * dm_kl[3];
            vj_ij[8] += R_0_1_2_1 * dm_kl[0];
            vj_ij[28] += R_0_1_2_1 * dm_kl[1];
            vj_ij[48] += R_0_1_2_1 * dm_kl[2];
            vj_ij[68] += R_0_1_2_1 * dm_kl[3];
            vj_ij[9] += R_0_1_3_0 * dm_kl[0];
            vj_ij[29] += R_0_1_3_0 * dm_kl[1];
            vj_ij[49] += R_0_1_3_0 * dm_kl[2];
            vj_ij[69] += R_0_1_3_0 * dm_kl[3];
            vj_ij[10] += R_0_2_0_0 * dm_kl[0];
            vj_ij[30] += R_0_2_0_0 * dm_kl[1];
            vj_ij[50] += R_0_2_0_0 * dm_kl[2];
            vj_ij[70] += R_0_2_0_0 * dm_kl[3];
            vj_ij[11] += R_0_2_0_1 * dm_kl[0];
            vj_ij[31] += R_0_2_0_1 * dm_kl[1];
            vj_ij[51] += R_0_2_0_1 * dm_kl[2];
            vj_ij[71] += R_0_2_0_1 * dm_kl[3];
            vj_ij[12] += R_0_2_0_2 * dm_kl[0];
            vj_ij[32] += R_0_2_0_2 * dm_kl[1];
            vj_ij[52] += R_0_2_0_2 * dm_kl[2];
            vj_ij[72] += R_0_2_0_2 * dm_kl[3];
            vj_ij[13] += R_0_2_1_0 * dm_kl[0];
            vj_ij[33] += R_0_2_1_0 * dm_kl[1];
            vj_ij[53] += R_0_2_1_0 * dm_kl[2];
            vj_ij[73] += R_0_2_1_0 * dm_kl[3];
            vj_ij[14] += R_0_2_1_1 * dm_kl[0];
            vj_ij[34] += R_0_2_1_1 * dm_kl[1];
            vj_ij[54] += R_0_2_1_1 * dm_kl[2];
            vj_ij[74] += R_0_2_1_1 * dm_kl[3];
            vj_ij[15] += R_0_2_2_0 * dm_kl[0];
            vj_ij[35] += R_0_2_2_0 * dm_kl[1];
            vj_ij[55] += R_0_2_2_0 * dm_kl[2];
            vj_ij[75] += R_0_2_2_0 * dm_kl[3];
            vj_ij[16] += R_0_3_0_0 * dm_kl[0];
            vj_ij[36] += R_0_3_0_0 * dm_kl[1];
            vj_ij[56] += R_0_3_0_0 * dm_kl[2];
            vj_ij[76] += R_0_3_0_0 * dm_kl[3];
            vj_ij[17] += R_0_3_0_1 * dm_kl[0];
            vj_ij[37] += R_0_3_0_1 * dm_kl[1];
            vj_ij[57] += R_0_3_0_1 * dm_kl[2];
            vj_ij[77] += R_0_3_0_1 * dm_kl[3];
            vj_ij[18] += R_0_3_1_0 * dm_kl[0];
            vj_ij[38] += R_0_3_1_0 * dm_kl[1];
            vj_ij[58] += R_0_3_1_0 * dm_kl[2];
            vj_ij[78] += R_0_3_1_0 * dm_kl[3];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[19] += R_0_4_0_0 * dm_kl[0];
            vj_ij[39] += R_0_4_0_0 * dm_kl[1];
            vj_ij[59] += R_0_4_0_0 * dm_kl[2];
            vj_ij[79] += R_0_4_0_0 * dm_kl[3];
            }
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
            int task_ij = task_ij0 + tx;
            double *vj_cache1 = vj_cache + 256;
#pragma unroll
            for (int n = 0; n < 20; ++n) {
                __syncthreads();
                vj_cache [thread_id] = vj_ij[n];
                vj_cache1[thread_id] = vj_ij[n+20];
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        vj_cache [thread_id] += vj_cache [thread_id + stride*16];
                        vj_cache1[thread_id] += vj_cache1[thread_id + stride*16];
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 20; ++n) {
                __syncthreads();
                for (int m = 0; m < 4; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+20*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 4; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 24; n += 16) {
        int kl = n / 6;
        int batch_kl = n - kl * 6;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 96 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*384+kl*96]);
            }
        }
    }
} }

// TILEX=48, TILEY=24
__global__ static
void md_j_4dm_4_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
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
    double vj_kl[2];
    double dm_kl[2];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 768;
    double *Rp_cache = Rq_cache + 1536;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 1184 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 1600; n += 256) {
        Rq_cache[n] = 0.;
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
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+384] = 1e5;
            Rq_cache[n+768] = 1e5;
            Rq_cache[n+1152] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 2) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 768; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 35 * min(remaining_n_dm, 2);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 35;
            int i = n - i_dm * 35;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[70];
        for (int ij = 0; ij < 70; ++ij) {
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
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[64];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[96];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[112];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[128];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[144];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[160];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[176];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[192];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[208];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[224];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[240];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[272];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[288];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[304];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[320];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[336];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[352];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[368];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[384];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[400];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[416];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[432];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[448];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[464];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[480];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[496];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[512];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[528];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[4] += R_0_0_0_4 * dm_kl[0];
            vj_ij[5] += R_0_0_1_0 * dm_kl[0];
            vj_ij[6] += R_0_0_1_1 * dm_kl[0];
            vj_ij[7] += R_0_0_1_2 * dm_kl[0];
            vj_ij[8] += R_0_0_1_3 * dm_kl[0];
            vj_ij[9] += R_0_0_2_0 * dm_kl[0];
            vj_ij[10] += R_0_0_2_1 * dm_kl[0];
            vj_ij[11] += R_0_0_2_2 * dm_kl[0];
            vj_ij[12] += R_0_0_3_0 * dm_kl[0];
            vj_ij[13] += R_0_0_3_1 * dm_kl[0];
            vj_ij[14] += R_0_0_4_0 * dm_kl[0];
            vj_ij[15] += R_0_1_0_0 * dm_kl[0];
            vj_ij[16] += R_0_1_0_1 * dm_kl[0];
            vj_ij[17] += R_0_1_0_2 * dm_kl[0];
            vj_ij[18] += R_0_1_0_3 * dm_kl[0];
            vj_ij[19] += R_0_1_1_0 * dm_kl[0];
            vj_ij[20] += R_0_1_1_1 * dm_kl[0];
            vj_ij[21] += R_0_1_1_2 * dm_kl[0];
            vj_ij[22] += R_0_1_2_0 * dm_kl[0];
            vj_ij[23] += R_0_1_2_1 * dm_kl[0];
            vj_ij[24] += R_0_1_3_0 * dm_kl[0];
            vj_ij[25] += R_0_2_0_0 * dm_kl[0];
            vj_ij[26] += R_0_2_0_1 * dm_kl[0];
            vj_ij[27] += R_0_2_0_2 * dm_kl[0];
            vj_ij[28] += R_0_2_1_0 * dm_kl[0];
            vj_ij[29] += R_0_2_1_1 * dm_kl[0];
            vj_ij[30] += R_0_2_2_0 * dm_kl[0];
            vj_ij[31] += R_0_3_0_0 * dm_kl[0];
            vj_ij[32] += R_0_3_0_1 * dm_kl[0];
            vj_ij[33] += R_0_3_1_0 * dm_kl[0];
            vj_ij[34] += R_0_4_0_0 * dm_kl[0];
            } else {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[560];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[576];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[592];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[608];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_0_4 * dm_ij_cache[624];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[640];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[96];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[656];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[112];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[672];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[128];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[688];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[704];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[160];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[720];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[176];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[736];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[192];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[752];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[208];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[768];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[224];
            vj_kl[1] += R_0_0_4_0 * dm_ij_cache[784];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[240];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[800];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[256];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[816];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[272];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[832];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[288];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[848];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[304];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[864];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[320];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[880];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[336];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[896];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[352];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[912];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[368];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[928];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[384];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[944];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[400];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[960];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[416];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[976];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[432];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[992];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[448];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[1008];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[464];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[1024];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[480];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[1040];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[496];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[1056];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[512];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[1072];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[528];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[1088];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[544];
            vj_kl[1] += R_0_4_0_0 * dm_ij_cache[1104];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*384+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[35] += gamma_inc[0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[36] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[37] += R_0_0_0_2 * dm_kl[1];
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[38] += R_0_0_0_3 * dm_kl[1];
            vj_ij[4] += R_0_0_0_4 * dm_kl[0];
            vj_ij[39] += R_0_0_0_4 * dm_kl[1];
            vj_ij[5] += R_0_0_1_0 * dm_kl[0];
            vj_ij[40] += R_0_0_1_0 * dm_kl[1];
            vj_ij[6] += R_0_0_1_1 * dm_kl[0];
            vj_ij[41] += R_0_0_1_1 * dm_kl[1];
            vj_ij[7] += R_0_0_1_2 * dm_kl[0];
            vj_ij[42] += R_0_0_1_2 * dm_kl[1];
            vj_ij[8] += R_0_0_1_3 * dm_kl[0];
            vj_ij[43] += R_0_0_1_3 * dm_kl[1];
            vj_ij[9] += R_0_0_2_0 * dm_kl[0];
            vj_ij[44] += R_0_0_2_0 * dm_kl[1];
            vj_ij[10] += R_0_0_2_1 * dm_kl[0];
            vj_ij[45] += R_0_0_2_1 * dm_kl[1];
            vj_ij[11] += R_0_0_2_2 * dm_kl[0];
            vj_ij[46] += R_0_0_2_2 * dm_kl[1];
            vj_ij[12] += R_0_0_3_0 * dm_kl[0];
            vj_ij[47] += R_0_0_3_0 * dm_kl[1];
            vj_ij[13] += R_0_0_3_1 * dm_kl[0];
            vj_ij[48] += R_0_0_3_1 * dm_kl[1];
            vj_ij[14] += R_0_0_4_0 * dm_kl[0];
            vj_ij[49] += R_0_0_4_0 * dm_kl[1];
            vj_ij[15] += R_0_1_0_0 * dm_kl[0];
            vj_ij[50] += R_0_1_0_0 * dm_kl[1];
            vj_ij[16] += R_0_1_0_1 * dm_kl[0];
            vj_ij[51] += R_0_1_0_1 * dm_kl[1];
            vj_ij[17] += R_0_1_0_2 * dm_kl[0];
            vj_ij[52] += R_0_1_0_2 * dm_kl[1];
            vj_ij[18] += R_0_1_0_3 * dm_kl[0];
            vj_ij[53] += R_0_1_0_3 * dm_kl[1];
            vj_ij[19] += R_0_1_1_0 * dm_kl[0];
            vj_ij[54] += R_0_1_1_0 * dm_kl[1];
            vj_ij[20] += R_0_1_1_1 * dm_kl[0];
            vj_ij[55] += R_0_1_1_1 * dm_kl[1];
            vj_ij[21] += R_0_1_1_2 * dm_kl[0];
            vj_ij[56] += R_0_1_1_2 * dm_kl[1];
            vj_ij[22] += R_0_1_2_0 * dm_kl[0];
            vj_ij[57] += R_0_1_2_0 * dm_kl[1];
            vj_ij[23] += R_0_1_2_1 * dm_kl[0];
            vj_ij[58] += R_0_1_2_1 * dm_kl[1];
            vj_ij[24] += R_0_1_3_0 * dm_kl[0];
            vj_ij[59] += R_0_1_3_0 * dm_kl[1];
            vj_ij[25] += R_0_2_0_0 * dm_kl[0];
            vj_ij[60] += R_0_2_0_0 * dm_kl[1];
            vj_ij[26] += R_0_2_0_1 * dm_kl[0];
            vj_ij[61] += R_0_2_0_1 * dm_kl[1];
            vj_ij[27] += R_0_2_0_2 * dm_kl[0];
            vj_ij[62] += R_0_2_0_2 * dm_kl[1];
            vj_ij[28] += R_0_2_1_0 * dm_kl[0];
            vj_ij[63] += R_0_2_1_0 * dm_kl[1];
            vj_ij[29] += R_0_2_1_1 * dm_kl[0];
            vj_ij[64] += R_0_2_1_1 * dm_kl[1];
            vj_ij[30] += R_0_2_2_0 * dm_kl[0];
            vj_ij[65] += R_0_2_2_0 * dm_kl[1];
            vj_ij[31] += R_0_3_0_0 * dm_kl[0];
            vj_ij[66] += R_0_3_0_0 * dm_kl[1];
            vj_ij[32] += R_0_3_0_1 * dm_kl[0];
            vj_ij[67] += R_0_3_0_1 * dm_kl[1];
            vj_ij[33] += R_0_3_1_0 * dm_kl[0];
            vj_ij[68] += R_0_3_1_0 * dm_kl[1];
            vj_ij[34] += R_0_4_0_0 * dm_kl[0];
            vj_ij[69] += R_0_4_0_0 * dm_kl[1];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 35; ++n) {
                __syncthreads();
                for (int m = 0; m < 2; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+35*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 2; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(2, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
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
            for (int m = 0; m < min(2, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*384+kl*384]);
            }
        }
    }
} }

// TILEX=48, TILEY=9
__global__ static
void md_j_4dm_4_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    int thread_id = sq_id;
    int *bas = envs.bas;
    int *pair_ij_loc = bounds.pair_ij_loc;
    int *pair_kl_loc = bounds.pair_kl_loc;
    int nbas = envs.nbas;
    double *env = envs.env;
    double vj_kl[2];
    double dm_kl[2];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 1152;
    double *Rp_cache = Rq_cache + 576;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 1184 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 640; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 144; n += 256) {
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
        } else {
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+144] = 1e5;
            Rq_cache[n+288] = 1e5;
            Rq_cache[n+432] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 2) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 1152; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 35 * min(remaining_n_dm, 2);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 35;
            int i = n - i_dm * 35;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[70];
        for (int ij = 0; ij < 70; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 9; ++batch_kl) {
            int task_kl0 = blockIdx.y * 144 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*9] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, 0, 256);
            if (remaining_n_dm == 1) {
            {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[64];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[96];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[112];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[128];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[144];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[160];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[176];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[192];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[208];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[224];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[240];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[272];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[288];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[304];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[320];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[336];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[352];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[368];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[384];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[400];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[416];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[432];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[448];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[464];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[480];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[496];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[512];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[528];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[48];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl[0] -= R_0_0_0_5 * dm_ij_cache[64];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[80];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[96];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[112];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl[0] -= R_0_0_1_4 * dm_ij_cache[128];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[144];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[160];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl[0] -= R_0_0_2_3 * dm_ij_cache[176];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[192];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl[0] -= R_0_0_3_2 * dm_ij_cache[208];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl[0] -= R_0_0_4_1 * dm_ij_cache[224];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[240];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[256];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[272];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl[0] -= R_0_1_0_4 * dm_ij_cache[288];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[304];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[320];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl[0] -= R_0_1_1_3 * dm_ij_cache[336];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[352];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl[0] -= R_0_1_2_2 * dm_ij_cache[368];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl[0] -= R_0_1_3_1 * dm_ij_cache[384];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[400];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[416];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl[0] -= R_0_2_0_3 * dm_ij_cache[432];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[448];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl[0] -= R_0_2_1_2 * dm_ij_cache[464];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl[0] -= R_0_2_2_1 * dm_ij_cache[480];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[496];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl[0] -= R_0_3_0_2 * dm_ij_cache[512];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl[0] -= R_0_3_1_1 * dm_ij_cache[528];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl[0] -= R_0_4_0_1 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+144] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl[0] -= R_0_0_1_4 * dm_ij_cache[64];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[96];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[112];
            vj_kl[0] -= R_0_0_2_3 * dm_ij_cache[128];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[144];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[160];
            vj_kl[0] -= R_0_0_3_2 * dm_ij_cache[176];
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[192];
            vj_kl[0] -= R_0_0_4_1 * dm_ij_cache[208];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl[0] -= R_0_0_5_0 * dm_ij_cache[224];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[256];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[272];
            vj_kl[0] -= R_0_1_1_3 * dm_ij_cache[288];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[304];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[320];
            vj_kl[0] -= R_0_1_2_2 * dm_ij_cache[336];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[352];
            vj_kl[0] -= R_0_1_3_1 * dm_ij_cache[368];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl[0] -= R_0_1_4_0 * dm_ij_cache[384];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[400];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[416];
            vj_kl[0] -= R_0_2_1_2 * dm_ij_cache[432];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[448];
            vj_kl[0] -= R_0_2_2_1 * dm_ij_cache[464];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl[0] -= R_0_2_3_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[496];
            vj_kl[0] -= R_0_3_1_1 * dm_ij_cache[512];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl[0] -= R_0_3_2_0 * dm_ij_cache[528];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl[0] -= R_0_4_1_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+288] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl[0] -= R_0_1_0_4 * dm_ij_cache[64];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[80];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[96];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[112];
            vj_kl[0] -= R_0_1_1_3 * dm_ij_cache[128];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[144];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[160];
            vj_kl[0] -= R_0_1_2_2 * dm_ij_cache[176];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[192];
            vj_kl[0] -= R_0_1_3_1 * dm_ij_cache[208];
            vj_kl[0] -= R_0_1_4_0 * dm_ij_cache[224];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[240];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[256];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[272];
            vj_kl[0] -= R_0_2_0_3 * dm_ij_cache[288];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[304];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[320];
            vj_kl[0] -= R_0_2_1_2 * dm_ij_cache[336];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[352];
            vj_kl[0] -= R_0_2_2_1 * dm_ij_cache[368];
            vj_kl[0] -= R_0_2_3_0 * dm_ij_cache[384];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[400];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[416];
            vj_kl[0] -= R_0_3_0_2 * dm_ij_cache[432];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[448];
            vj_kl[0] -= R_0_3_1_1 * dm_ij_cache[464];
            vj_kl[0] -= R_0_3_2_0 * dm_ij_cache[480];
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[496];
            vj_kl[0] -= R_0_4_0_1 * dm_ij_cache[512];
            vj_kl[0] -= R_0_4_1_0 * dm_ij_cache[528];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl[0] -= R_0_5_0_0 * dm_ij_cache[544];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+432] += vj_kl[m];
            } }
            }{
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[4] += R_0_0_0_4 * dm_kl[0];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[5] += R_0_0_1_0 * dm_kl[0];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[6] += R_0_0_1_1 * dm_kl[0];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[7] += R_0_0_1_2 * dm_kl[0];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[8] += R_0_0_1_3 * dm_kl[0];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_ij[9] += R_0_0_2_0 * dm_kl[0];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[10] += R_0_0_2_1 * dm_kl[0];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[11] += R_0_0_2_2 * dm_kl[0];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[12] += R_0_0_3_0 * dm_kl[0];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[13] += R_0_0_3_1 * dm_kl[0];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[14] += R_0_0_4_0 * dm_kl[0];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_ij[15] += R_0_1_0_0 * dm_kl[0];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[16] += R_0_1_0_1 * dm_kl[0];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[17] += R_0_1_0_2 * dm_kl[0];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[18] += R_0_1_0_3 * dm_kl[0];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[19] += R_0_1_1_0 * dm_kl[0];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[20] += R_0_1_1_1 * dm_kl[0];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[21] += R_0_1_1_2 * dm_kl[0];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[22] += R_0_1_2_0 * dm_kl[0];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[23] += R_0_1_2_1 * dm_kl[0];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[24] += R_0_1_3_0 * dm_kl[0];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_ij[25] += R_0_2_0_0 * dm_kl[0];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[26] += R_0_2_0_1 * dm_kl[0];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[27] += R_0_2_0_2 * dm_kl[0];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[28] += R_0_2_1_0 * dm_kl[0];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[29] += R_0_2_1_1 * dm_kl[0];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[30] += R_0_2_2_0 * dm_kl[0];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[31] += R_0_3_0_0 * dm_kl[0];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[32] += R_0_3_0_1 * dm_kl[0];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[33] += R_0_3_1_0 * dm_kl[0];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[34] += R_0_4_0_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+1];
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[3] += R_0_0_0_4 * dm_kl[0];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_ij[4] += R_0_0_0_5 * dm_kl[0];
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[7] += R_0_0_1_3 * dm_kl[0];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_ij[8] += R_0_0_1_4 * dm_kl[0];
            vj_ij[9] += R_0_0_2_1 * dm_kl[0];
            vj_ij[10] += R_0_0_2_2 * dm_kl[0];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_ij[11] += R_0_0_2_3 * dm_kl[0];
            vj_ij[12] += R_0_0_3_1 * dm_kl[0];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_ij[13] += R_0_0_3_2 * dm_kl[0];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_ij[14] += R_0_0_4_1 * dm_kl[0];
            vj_ij[15] += R_0_1_0_1 * dm_kl[0];
            vj_ij[16] += R_0_1_0_2 * dm_kl[0];
            vj_ij[17] += R_0_1_0_3 * dm_kl[0];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_ij[18] += R_0_1_0_4 * dm_kl[0];
            vj_ij[19] += R_0_1_1_1 * dm_kl[0];
            vj_ij[20] += R_0_1_1_2 * dm_kl[0];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_ij[21] += R_0_1_1_3 * dm_kl[0];
            vj_ij[22] += R_0_1_2_1 * dm_kl[0];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_ij[23] += R_0_1_2_2 * dm_kl[0];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_ij[24] += R_0_1_3_1 * dm_kl[0];
            vj_ij[25] += R_0_2_0_1 * dm_kl[0];
            vj_ij[26] += R_0_2_0_2 * dm_kl[0];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_ij[27] += R_0_2_0_3 * dm_kl[0];
            vj_ij[28] += R_0_2_1_1 * dm_kl[0];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_ij[29] += R_0_2_1_2 * dm_kl[0];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_ij[30] += R_0_2_2_1 * dm_kl[0];
            vj_ij[31] += R_0_3_0_1 * dm_kl[0];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_ij[32] += R_0_3_0_2 * dm_kl[0];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_ij[33] += R_0_3_1_1 * dm_kl[0];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_ij[34] += R_0_4_0_1 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+2];
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[3] += R_0_0_1_3 * dm_kl[0];
            vj_ij[4] += R_0_0_1_4 * dm_kl[0];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[6] += R_0_0_2_1 * dm_kl[0];
            vj_ij[7] += R_0_0_2_2 * dm_kl[0];
            vj_ij[8] += R_0_0_2_3 * dm_kl[0];
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[10] += R_0_0_3_1 * dm_kl[0];
            vj_ij[11] += R_0_0_3_2 * dm_kl[0];
            vj_ij[12] += R_0_0_4_0 * dm_kl[0];
            vj_ij[13] += R_0_0_4_1 * dm_kl[0];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_ij[14] += R_0_0_5_0 * dm_kl[0];
            vj_ij[15] += R_0_1_1_0 * dm_kl[0];
            vj_ij[16] += R_0_1_1_1 * dm_kl[0];
            vj_ij[17] += R_0_1_1_2 * dm_kl[0];
            vj_ij[18] += R_0_1_1_3 * dm_kl[0];
            vj_ij[19] += R_0_1_2_0 * dm_kl[0];
            vj_ij[20] += R_0_1_2_1 * dm_kl[0];
            vj_ij[21] += R_0_1_2_2 * dm_kl[0];
            vj_ij[22] += R_0_1_3_0 * dm_kl[0];
            vj_ij[23] += R_0_1_3_1 * dm_kl[0];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_ij[24] += R_0_1_4_0 * dm_kl[0];
            vj_ij[25] += R_0_2_1_0 * dm_kl[0];
            vj_ij[26] += R_0_2_1_1 * dm_kl[0];
            vj_ij[27] += R_0_2_1_2 * dm_kl[0];
            vj_ij[28] += R_0_2_2_0 * dm_kl[0];
            vj_ij[29] += R_0_2_2_1 * dm_kl[0];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_ij[30] += R_0_2_3_0 * dm_kl[0];
            vj_ij[31] += R_0_3_1_0 * dm_kl[0];
            vj_ij[32] += R_0_3_1_1 * dm_kl[0];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_ij[33] += R_0_3_2_0 * dm_kl[0];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_ij[34] += R_0_4_1_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+3];
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[3] += R_0_1_0_3 * dm_kl[0];
            vj_ij[4] += R_0_1_0_4 * dm_kl[0];
            vj_ij[5] += R_0_1_1_0 * dm_kl[0];
            vj_ij[6] += R_0_1_1_1 * dm_kl[0];
            vj_ij[7] += R_0_1_1_2 * dm_kl[0];
            vj_ij[8] += R_0_1_1_3 * dm_kl[0];
            vj_ij[9] += R_0_1_2_0 * dm_kl[0];
            vj_ij[10] += R_0_1_2_1 * dm_kl[0];
            vj_ij[11] += R_0_1_2_2 * dm_kl[0];
            vj_ij[12] += R_0_1_3_0 * dm_kl[0];
            vj_ij[13] += R_0_1_3_1 * dm_kl[0];
            vj_ij[14] += R_0_1_4_0 * dm_kl[0];
            vj_ij[15] += R_0_2_0_0 * dm_kl[0];
            vj_ij[16] += R_0_2_0_1 * dm_kl[0];
            vj_ij[17] += R_0_2_0_2 * dm_kl[0];
            vj_ij[18] += R_0_2_0_3 * dm_kl[0];
            vj_ij[19] += R_0_2_1_0 * dm_kl[0];
            vj_ij[20] += R_0_2_1_1 * dm_kl[0];
            vj_ij[21] += R_0_2_1_2 * dm_kl[0];
            vj_ij[22] += R_0_2_2_0 * dm_kl[0];
            vj_ij[23] += R_0_2_2_1 * dm_kl[0];
            vj_ij[24] += R_0_2_3_0 * dm_kl[0];
            vj_ij[25] += R_0_3_0_0 * dm_kl[0];
            vj_ij[26] += R_0_3_0_1 * dm_kl[0];
            vj_ij[27] += R_0_3_0_2 * dm_kl[0];
            vj_ij[28] += R_0_3_1_0 * dm_kl[0];
            vj_ij[29] += R_0_3_1_1 * dm_kl[0];
            vj_ij[30] += R_0_3_2_0 * dm_kl[0];
            vj_ij[31] += R_0_4_0_0 * dm_kl[0];
            vj_ij[32] += R_0_4_0_1 * dm_kl[0];
            vj_ij[33] += R_0_4_1_0 * dm_kl[0];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_ij[34] += R_0_5_0_0 * dm_kl[0];
            }
            } else {
            {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[560];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[576];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[592];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[608];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_0_4 * dm_ij_cache[624];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[640];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[96];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[656];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[112];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[672];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[128];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[688];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[144];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[704];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[160];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[720];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[176];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[736];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[192];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[752];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[208];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[768];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[224];
            vj_kl[1] += R_0_0_4_0 * dm_ij_cache[784];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[240];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[800];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[256];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[816];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[272];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[832];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[288];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[848];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[304];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[864];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[320];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[880];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[336];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[896];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[352];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[912];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[368];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[928];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[384];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[944];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[400];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[960];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[416];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[976];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[432];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[992];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[448];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[1008];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[464];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[1024];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[480];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[1040];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[496];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[1056];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[512];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[1072];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[528];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[1088];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[544];
            vj_kl[1] += R_0_4_0_0 * dm_ij_cache[1104];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[560];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[576];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[592];
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_0_4 * dm_ij_cache[608];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl[0] -= R_0_0_0_5 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_0_5 * dm_ij_cache[624];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[640];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[96];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[656];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[112];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[672];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl[0] -= R_0_0_1_4 * dm_ij_cache[128];
            vj_kl[1] -= R_0_0_1_4 * dm_ij_cache[688];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[144];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[704];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[160];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[720];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl[0] -= R_0_0_2_3 * dm_ij_cache[176];
            vj_kl[1] -= R_0_0_2_3 * dm_ij_cache[736];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[192];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[752];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl[0] -= R_0_0_3_2 * dm_ij_cache[208];
            vj_kl[1] -= R_0_0_3_2 * dm_ij_cache[768];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl[0] -= R_0_0_4_1 * dm_ij_cache[224];
            vj_kl[1] -= R_0_0_4_1 * dm_ij_cache[784];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[240];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[800];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[256];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[816];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[272];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[832];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl[0] -= R_0_1_0_4 * dm_ij_cache[288];
            vj_kl[1] -= R_0_1_0_4 * dm_ij_cache[848];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[304];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[864];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[320];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[880];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl[0] -= R_0_1_1_3 * dm_ij_cache[336];
            vj_kl[1] -= R_0_1_1_3 * dm_ij_cache[896];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[352];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[912];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl[0] -= R_0_1_2_2 * dm_ij_cache[368];
            vj_kl[1] -= R_0_1_2_2 * dm_ij_cache[928];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl[0] -= R_0_1_3_1 * dm_ij_cache[384];
            vj_kl[1] -= R_0_1_3_1 * dm_ij_cache[944];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[400];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[960];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[416];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[976];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl[0] -= R_0_2_0_3 * dm_ij_cache[432];
            vj_kl[1] -= R_0_2_0_3 * dm_ij_cache[992];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[448];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[1008];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl[0] -= R_0_2_1_2 * dm_ij_cache[464];
            vj_kl[1] -= R_0_2_1_2 * dm_ij_cache[1024];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl[0] -= R_0_2_2_1 * dm_ij_cache[480];
            vj_kl[1] -= R_0_2_2_1 * dm_ij_cache[1040];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[496];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[1056];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl[0] -= R_0_3_0_2 * dm_ij_cache[512];
            vj_kl[1] -= R_0_3_0_2 * dm_ij_cache[1072];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl[0] -= R_0_3_1_1 * dm_ij_cache[528];
            vj_kl[1] -= R_0_3_1_1 * dm_ij_cache[1088];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl[0] -= R_0_4_0_1 * dm_ij_cache[544];
            vj_kl[1] -= R_0_4_0_1 * dm_ij_cache[1104];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+144] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[576];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[592];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[48];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[608];
            vj_kl[0] -= R_0_0_1_4 * dm_ij_cache[64];
            vj_kl[1] -= R_0_0_1_4 * dm_ij_cache[624];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[640];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[96];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[656];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[112];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[672];
            vj_kl[0] -= R_0_0_2_3 * dm_ij_cache[128];
            vj_kl[1] -= R_0_0_2_3 * dm_ij_cache[688];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[704];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[160];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[720];
            vj_kl[0] -= R_0_0_3_2 * dm_ij_cache[176];
            vj_kl[1] -= R_0_0_3_2 * dm_ij_cache[736];
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[192];
            vj_kl[1] -= R_0_0_4_0 * dm_ij_cache[752];
            vj_kl[0] -= R_0_0_4_1 * dm_ij_cache[208];
            vj_kl[1] -= R_0_0_4_1 * dm_ij_cache[768];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl[0] -= R_0_0_5_0 * dm_ij_cache[224];
            vj_kl[1] -= R_0_0_5_0 * dm_ij_cache[784];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[240];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[800];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[256];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[816];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[272];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[832];
            vj_kl[0] -= R_0_1_1_3 * dm_ij_cache[288];
            vj_kl[1] -= R_0_1_1_3 * dm_ij_cache[848];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[304];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[864];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[320];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[880];
            vj_kl[0] -= R_0_1_2_2 * dm_ij_cache[336];
            vj_kl[1] -= R_0_1_2_2 * dm_ij_cache[896];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[352];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[912];
            vj_kl[0] -= R_0_1_3_1 * dm_ij_cache[368];
            vj_kl[1] -= R_0_1_3_1 * dm_ij_cache[928];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl[0] -= R_0_1_4_0 * dm_ij_cache[384];
            vj_kl[1] -= R_0_1_4_0 * dm_ij_cache[944];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[400];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[960];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[416];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[976];
            vj_kl[0] -= R_0_2_1_2 * dm_ij_cache[432];
            vj_kl[1] -= R_0_2_1_2 * dm_ij_cache[992];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[448];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[1008];
            vj_kl[0] -= R_0_2_2_1 * dm_ij_cache[464];
            vj_kl[1] -= R_0_2_2_1 * dm_ij_cache[1024];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl[0] -= R_0_2_3_0 * dm_ij_cache[480];
            vj_kl[1] -= R_0_2_3_0 * dm_ij_cache[1040];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[496];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[1056];
            vj_kl[0] -= R_0_3_1_1 * dm_ij_cache[512];
            vj_kl[1] -= R_0_3_1_1 * dm_ij_cache[1072];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl[0] -= R_0_3_2_0 * dm_ij_cache[528];
            vj_kl[1] -= R_0_3_2_0 * dm_ij_cache[1088];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl[0] -= R_0_4_1_0 * dm_ij_cache[544];
            vj_kl[1] -= R_0_4_1_0 * dm_ij_cache[1104];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+288] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[560];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[576];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[592];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[48];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[608];
            vj_kl[0] -= R_0_1_0_4 * dm_ij_cache[64];
            vj_kl[1] -= R_0_1_0_4 * dm_ij_cache[624];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[80];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[640];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[96];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[656];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[112];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[672];
            vj_kl[0] -= R_0_1_1_3 * dm_ij_cache[128];
            vj_kl[1] -= R_0_1_1_3 * dm_ij_cache[688];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[144];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[704];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[160];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[720];
            vj_kl[0] -= R_0_1_2_2 * dm_ij_cache[176];
            vj_kl[1] -= R_0_1_2_2 * dm_ij_cache[736];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[192];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[752];
            vj_kl[0] -= R_0_1_3_1 * dm_ij_cache[208];
            vj_kl[1] -= R_0_1_3_1 * dm_ij_cache[768];
            vj_kl[0] -= R_0_1_4_0 * dm_ij_cache[224];
            vj_kl[1] -= R_0_1_4_0 * dm_ij_cache[784];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[240];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[800];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[256];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[816];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[272];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[832];
            vj_kl[0] -= R_0_2_0_3 * dm_ij_cache[288];
            vj_kl[1] -= R_0_2_0_3 * dm_ij_cache[848];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[304];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[864];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[320];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[880];
            vj_kl[0] -= R_0_2_1_2 * dm_ij_cache[336];
            vj_kl[1] -= R_0_2_1_2 * dm_ij_cache[896];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[352];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[912];
            vj_kl[0] -= R_0_2_2_1 * dm_ij_cache[368];
            vj_kl[1] -= R_0_2_2_1 * dm_ij_cache[928];
            vj_kl[0] -= R_0_2_3_0 * dm_ij_cache[384];
            vj_kl[1] -= R_0_2_3_0 * dm_ij_cache[944];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[400];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[960];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[416];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[976];
            vj_kl[0] -= R_0_3_0_2 * dm_ij_cache[432];
            vj_kl[1] -= R_0_3_0_2 * dm_ij_cache[992];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[448];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[1008];
            vj_kl[0] -= R_0_3_1_1 * dm_ij_cache[464];
            vj_kl[1] -= R_0_3_1_1 * dm_ij_cache[1024];
            vj_kl[0] -= R_0_3_2_0 * dm_ij_cache[480];
            vj_kl[1] -= R_0_3_2_0 * dm_ij_cache[1040];
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[496];
            vj_kl[1] -= R_0_4_0_0 * dm_ij_cache[1056];
            vj_kl[0] -= R_0_4_0_1 * dm_ij_cache[512];
            vj_kl[1] -= R_0_4_0_1 * dm_ij_cache[1072];
            vj_kl[0] -= R_0_4_1_0 * dm_ij_cache[528];
            vj_kl[1] -= R_0_4_1_0 * dm_ij_cache[1088];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl[0] -= R_0_5_0_0 * dm_ij_cache[544];
            vj_kl[1] -= R_0_5_0_0 * dm_ij_cache[1104];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*576+432] += vj_kl[m];
            } }
            }{
            for (int m = 0; m < 2; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[35] += gamma_inc[0*256] * dm_kl[1];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[36] += R_0_0_0_1 * dm_kl[1];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[37] += R_0_0_0_2 * dm_kl[1];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[38] += R_0_0_0_3 * dm_kl[1];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[4] += R_0_0_0_4 * dm_kl[0];
            vj_ij[39] += R_0_0_0_4 * dm_kl[1];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[5] += R_0_0_1_0 * dm_kl[0];
            vj_ij[40] += R_0_0_1_0 * dm_kl[1];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[6] += R_0_0_1_1 * dm_kl[0];
            vj_ij[41] += R_0_0_1_1 * dm_kl[1];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[7] += R_0_0_1_2 * dm_kl[0];
            vj_ij[42] += R_0_0_1_2 * dm_kl[1];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[8] += R_0_0_1_3 * dm_kl[0];
            vj_ij[43] += R_0_0_1_3 * dm_kl[1];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_ij[9] += R_0_0_2_0 * dm_kl[0];
            vj_ij[44] += R_0_0_2_0 * dm_kl[1];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[10] += R_0_0_2_1 * dm_kl[0];
            vj_ij[45] += R_0_0_2_1 * dm_kl[1];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[11] += R_0_0_2_2 * dm_kl[0];
            vj_ij[46] += R_0_0_2_2 * dm_kl[1];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[12] += R_0_0_3_0 * dm_kl[0];
            vj_ij[47] += R_0_0_3_0 * dm_kl[1];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[13] += R_0_0_3_1 * dm_kl[0];
            vj_ij[48] += R_0_0_3_1 * dm_kl[1];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[14] += R_0_0_4_0 * dm_kl[0];
            vj_ij[49] += R_0_0_4_0 * dm_kl[1];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_ij[15] += R_0_1_0_0 * dm_kl[0];
            vj_ij[50] += R_0_1_0_0 * dm_kl[1];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[16] += R_0_1_0_1 * dm_kl[0];
            vj_ij[51] += R_0_1_0_1 * dm_kl[1];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[17] += R_0_1_0_2 * dm_kl[0];
            vj_ij[52] += R_0_1_0_2 * dm_kl[1];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[18] += R_0_1_0_3 * dm_kl[0];
            vj_ij[53] += R_0_1_0_3 * dm_kl[1];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[19] += R_0_1_1_0 * dm_kl[0];
            vj_ij[54] += R_0_1_1_0 * dm_kl[1];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[20] += R_0_1_1_1 * dm_kl[0];
            vj_ij[55] += R_0_1_1_1 * dm_kl[1];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[21] += R_0_1_1_2 * dm_kl[0];
            vj_ij[56] += R_0_1_1_2 * dm_kl[1];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[22] += R_0_1_2_0 * dm_kl[0];
            vj_ij[57] += R_0_1_2_0 * dm_kl[1];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[23] += R_0_1_2_1 * dm_kl[0];
            vj_ij[58] += R_0_1_2_1 * dm_kl[1];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[24] += R_0_1_3_0 * dm_kl[0];
            vj_ij[59] += R_0_1_3_0 * dm_kl[1];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_ij[25] += R_0_2_0_0 * dm_kl[0];
            vj_ij[60] += R_0_2_0_0 * dm_kl[1];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[26] += R_0_2_0_1 * dm_kl[0];
            vj_ij[61] += R_0_2_0_1 * dm_kl[1];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[27] += R_0_2_0_2 * dm_kl[0];
            vj_ij[62] += R_0_2_0_2 * dm_kl[1];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[28] += R_0_2_1_0 * dm_kl[0];
            vj_ij[63] += R_0_2_1_0 * dm_kl[1];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[29] += R_0_2_1_1 * dm_kl[0];
            vj_ij[64] += R_0_2_1_1 * dm_kl[1];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[30] += R_0_2_2_0 * dm_kl[0];
            vj_ij[65] += R_0_2_2_0 * dm_kl[1];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[31] += R_0_3_0_0 * dm_kl[0];
            vj_ij[66] += R_0_3_0_0 * dm_kl[1];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[32] += R_0_3_0_1 * dm_kl[0];
            vj_ij[67] += R_0_3_0_1 * dm_kl[1];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[33] += R_0_3_1_0 * dm_kl[0];
            vj_ij[68] += R_0_3_1_0 * dm_kl[1];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[34] += R_0_4_0_0 * dm_kl[0];
            vj_ij[69] += R_0_4_0_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1];
            }
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[35] += R_0_0_0_1 * dm_kl[1];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[36] += R_0_0_0_2 * dm_kl[1];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            vj_ij[37] += R_0_0_0_3 * dm_kl[1];
            vj_ij[3] += R_0_0_0_4 * dm_kl[0];
            vj_ij[38] += R_0_0_0_4 * dm_kl[1];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_ij[4] += R_0_0_0_5 * dm_kl[0];
            vj_ij[39] += R_0_0_0_5 * dm_kl[1];
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[40] += R_0_0_1_1 * dm_kl[1];
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[41] += R_0_0_1_2 * dm_kl[1];
            vj_ij[7] += R_0_0_1_3 * dm_kl[0];
            vj_ij[42] += R_0_0_1_3 * dm_kl[1];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_ij[8] += R_0_0_1_4 * dm_kl[0];
            vj_ij[43] += R_0_0_1_4 * dm_kl[1];
            vj_ij[9] += R_0_0_2_1 * dm_kl[0];
            vj_ij[44] += R_0_0_2_1 * dm_kl[1];
            vj_ij[10] += R_0_0_2_2 * dm_kl[0];
            vj_ij[45] += R_0_0_2_2 * dm_kl[1];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_ij[11] += R_0_0_2_3 * dm_kl[0];
            vj_ij[46] += R_0_0_2_3 * dm_kl[1];
            vj_ij[12] += R_0_0_3_1 * dm_kl[0];
            vj_ij[47] += R_0_0_3_1 * dm_kl[1];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_ij[13] += R_0_0_3_2 * dm_kl[0];
            vj_ij[48] += R_0_0_3_2 * dm_kl[1];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_ij[14] += R_0_0_4_1 * dm_kl[0];
            vj_ij[49] += R_0_0_4_1 * dm_kl[1];
            vj_ij[15] += R_0_1_0_1 * dm_kl[0];
            vj_ij[50] += R_0_1_0_1 * dm_kl[1];
            vj_ij[16] += R_0_1_0_2 * dm_kl[0];
            vj_ij[51] += R_0_1_0_2 * dm_kl[1];
            vj_ij[17] += R_0_1_0_3 * dm_kl[0];
            vj_ij[52] += R_0_1_0_3 * dm_kl[1];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_ij[18] += R_0_1_0_4 * dm_kl[0];
            vj_ij[53] += R_0_1_0_4 * dm_kl[1];
            vj_ij[19] += R_0_1_1_1 * dm_kl[0];
            vj_ij[54] += R_0_1_1_1 * dm_kl[1];
            vj_ij[20] += R_0_1_1_2 * dm_kl[0];
            vj_ij[55] += R_0_1_1_2 * dm_kl[1];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_ij[21] += R_0_1_1_3 * dm_kl[0];
            vj_ij[56] += R_0_1_1_3 * dm_kl[1];
            vj_ij[22] += R_0_1_2_1 * dm_kl[0];
            vj_ij[57] += R_0_1_2_1 * dm_kl[1];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_ij[23] += R_0_1_2_2 * dm_kl[0];
            vj_ij[58] += R_0_1_2_2 * dm_kl[1];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_ij[24] += R_0_1_3_1 * dm_kl[0];
            vj_ij[59] += R_0_1_3_1 * dm_kl[1];
            vj_ij[25] += R_0_2_0_1 * dm_kl[0];
            vj_ij[60] += R_0_2_0_1 * dm_kl[1];
            vj_ij[26] += R_0_2_0_2 * dm_kl[0];
            vj_ij[61] += R_0_2_0_2 * dm_kl[1];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_ij[27] += R_0_2_0_3 * dm_kl[0];
            vj_ij[62] += R_0_2_0_3 * dm_kl[1];
            vj_ij[28] += R_0_2_1_1 * dm_kl[0];
            vj_ij[63] += R_0_2_1_1 * dm_kl[1];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_ij[29] += R_0_2_1_2 * dm_kl[0];
            vj_ij[64] += R_0_2_1_2 * dm_kl[1];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_ij[30] += R_0_2_2_1 * dm_kl[0];
            vj_ij[65] += R_0_2_2_1 * dm_kl[1];
            vj_ij[31] += R_0_3_0_1 * dm_kl[0];
            vj_ij[66] += R_0_3_0_1 * dm_kl[1];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_ij[32] += R_0_3_0_2 * dm_kl[0];
            vj_ij[67] += R_0_3_0_2 * dm_kl[1];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_ij[33] += R_0_3_1_1 * dm_kl[0];
            vj_ij[68] += R_0_3_1_1 * dm_kl[1];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_ij[34] += R_0_4_0_1 * dm_kl[0];
            vj_ij[69] += R_0_4_0_1 * dm_kl[1];
            for (int m = 0; m < 2; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2];
            }
            vj_ij[0] += R_0_0_1_0 * dm_kl[0];
            vj_ij[35] += R_0_0_1_0 * dm_kl[1];
            vj_ij[1] += R_0_0_1_1 * dm_kl[0];
            vj_ij[36] += R_0_0_1_1 * dm_kl[1];
            vj_ij[2] += R_0_0_1_2 * dm_kl[0];
            vj_ij[37] += R_0_0_1_2 * dm_kl[1];
            vj_ij[3] += R_0_0_1_3 * dm_kl[0];
            vj_ij[38] += R_0_0_1_3 * dm_kl[1];
            vj_ij[4] += R_0_0_1_4 * dm_kl[0];
            vj_ij[39] += R_0_0_1_4 * dm_kl[1];
            vj_ij[5] += R_0_0_2_0 * dm_kl[0];
            vj_ij[40] += R_0_0_2_0 * dm_kl[1];
            vj_ij[6] += R_0_0_2_1 * dm_kl[0];
            vj_ij[41] += R_0_0_2_1 * dm_kl[1];
            vj_ij[7] += R_0_0_2_2 * dm_kl[0];
            vj_ij[42] += R_0_0_2_2 * dm_kl[1];
            vj_ij[8] += R_0_0_2_3 * dm_kl[0];
            vj_ij[43] += R_0_0_2_3 * dm_kl[1];
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[44] += R_0_0_3_0 * dm_kl[1];
            vj_ij[10] += R_0_0_3_1 * dm_kl[0];
            vj_ij[45] += R_0_0_3_1 * dm_kl[1];
            vj_ij[11] += R_0_0_3_2 * dm_kl[0];
            vj_ij[46] += R_0_0_3_2 * dm_kl[1];
            vj_ij[12] += R_0_0_4_0 * dm_kl[0];
            vj_ij[47] += R_0_0_4_0 * dm_kl[1];
            vj_ij[13] += R_0_0_4_1 * dm_kl[0];
            vj_ij[48] += R_0_0_4_1 * dm_kl[1];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_ij[14] += R_0_0_5_0 * dm_kl[0];
            vj_ij[49] += R_0_0_5_0 * dm_kl[1];
            vj_ij[15] += R_0_1_1_0 * dm_kl[0];
            vj_ij[50] += R_0_1_1_0 * dm_kl[1];
            vj_ij[16] += R_0_1_1_1 * dm_kl[0];
            vj_ij[51] += R_0_1_1_1 * dm_kl[1];
            vj_ij[17] += R_0_1_1_2 * dm_kl[0];
            vj_ij[52] += R_0_1_1_2 * dm_kl[1];
            vj_ij[18] += R_0_1_1_3 * dm_kl[0];
            vj_ij[53] += R_0_1_1_3 * dm_kl[1];
            vj_ij[19] += R_0_1_2_0 * dm_kl[0];
            vj_ij[54] += R_0_1_2_0 * dm_kl[1];
            vj_ij[20] += R_0_1_2_1 * dm_kl[0];
            vj_ij[55] += R_0_1_2_1 * dm_kl[1];
            vj_ij[21] += R_0_1_2_2 * dm_kl[0];
            vj_ij[56] += R_0_1_2_2 * dm_kl[1];
            vj_ij[22] += R_0_1_3_0 * dm_kl[0];
            vj_ij[57] += R_0_1_3_0 * dm_kl[1];
            vj_ij[23] += R_0_1_3_1 * dm_kl[0];
            vj_ij[58] += R_0_1_3_1 * dm_kl[1];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_ij[24] += R_0_1_4_0 * dm_kl[0];
            vj_ij[59] += R_0_1_4_0 * dm_kl[1];
            vj_ij[25] += R_0_2_1_0 * dm_kl[0];
            vj_ij[60] += R_0_2_1_0 * dm_kl[1];
            vj_ij[26] += R_0_2_1_1 * dm_kl[0];
            vj_ij[61] += R_0_2_1_1 * dm_kl[1];
            vj_ij[27] += R_0_2_1_2 * dm_kl[0];
            vj_ij[62] += R_0_2_1_2 * dm_kl[1];
            vj_ij[28] += R_0_2_2_0 * dm_kl[0];
            vj_ij[63] += R_0_2_2_0 * dm_kl[1];
            vj_ij[29] += R_0_2_2_1 * dm_kl[0];
            vj_ij[64] += R_0_2_2_1 * dm_kl[1];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_ij[30] += R_0_2_3_0 * dm_kl[0];
            vj_ij[65] += R_0_2_3_0 * dm_kl[1];
            vj_ij[31] += R_0_3_1_0 * dm_kl[0];
            vj_ij[66] += R_0_3_1_0 * dm_kl[1];
            vj_ij[32] += R_0_3_1_1 * dm_kl[0];
            vj_ij[67] += R_0_3_1_1 * dm_kl[1];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_ij[33] += R_0_3_2_0 * dm_kl[0];
            vj_ij[68] += R_0_3_2_0 * dm_kl[1];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_ij[34] += R_0_4_1_0 * dm_kl[0];
            vj_ij[69] += R_0_4_1_0 * dm_kl[1];
            for (int m = 0; m < 2; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3];
            }
            vj_ij[0] += R_0_1_0_0 * dm_kl[0];
            vj_ij[35] += R_0_1_0_0 * dm_kl[1];
            vj_ij[1] += R_0_1_0_1 * dm_kl[0];
            vj_ij[36] += R_0_1_0_1 * dm_kl[1];
            vj_ij[2] += R_0_1_0_2 * dm_kl[0];
            vj_ij[37] += R_0_1_0_2 * dm_kl[1];
            vj_ij[3] += R_0_1_0_3 * dm_kl[0];
            vj_ij[38] += R_0_1_0_3 * dm_kl[1];
            vj_ij[4] += R_0_1_0_4 * dm_kl[0];
            vj_ij[39] += R_0_1_0_4 * dm_kl[1];
            vj_ij[5] += R_0_1_1_0 * dm_kl[0];
            vj_ij[40] += R_0_1_1_0 * dm_kl[1];
            vj_ij[6] += R_0_1_1_1 * dm_kl[0];
            vj_ij[41] += R_0_1_1_1 * dm_kl[1];
            vj_ij[7] += R_0_1_1_2 * dm_kl[0];
            vj_ij[42] += R_0_1_1_2 * dm_kl[1];
            vj_ij[8] += R_0_1_1_3 * dm_kl[0];
            vj_ij[43] += R_0_1_1_3 * dm_kl[1];
            vj_ij[9] += R_0_1_2_0 * dm_kl[0];
            vj_ij[44] += R_0_1_2_0 * dm_kl[1];
            vj_ij[10] += R_0_1_2_1 * dm_kl[0];
            vj_ij[45] += R_0_1_2_1 * dm_kl[1];
            vj_ij[11] += R_0_1_2_2 * dm_kl[0];
            vj_ij[46] += R_0_1_2_2 * dm_kl[1];
            vj_ij[12] += R_0_1_3_0 * dm_kl[0];
            vj_ij[47] += R_0_1_3_0 * dm_kl[1];
            vj_ij[13] += R_0_1_3_1 * dm_kl[0];
            vj_ij[48] += R_0_1_3_1 * dm_kl[1];
            vj_ij[14] += R_0_1_4_0 * dm_kl[0];
            vj_ij[49] += R_0_1_4_0 * dm_kl[1];
            vj_ij[15] += R_0_2_0_0 * dm_kl[0];
            vj_ij[50] += R_0_2_0_0 * dm_kl[1];
            vj_ij[16] += R_0_2_0_1 * dm_kl[0];
            vj_ij[51] += R_0_2_0_1 * dm_kl[1];
            vj_ij[17] += R_0_2_0_2 * dm_kl[0];
            vj_ij[52] += R_0_2_0_2 * dm_kl[1];
            vj_ij[18] += R_0_2_0_3 * dm_kl[0];
            vj_ij[53] += R_0_2_0_3 * dm_kl[1];
            vj_ij[19] += R_0_2_1_0 * dm_kl[0];
            vj_ij[54] += R_0_2_1_0 * dm_kl[1];
            vj_ij[20] += R_0_2_1_1 * dm_kl[0];
            vj_ij[55] += R_0_2_1_1 * dm_kl[1];
            vj_ij[21] += R_0_2_1_2 * dm_kl[0];
            vj_ij[56] += R_0_2_1_2 * dm_kl[1];
            vj_ij[22] += R_0_2_2_0 * dm_kl[0];
            vj_ij[57] += R_0_2_2_0 * dm_kl[1];
            vj_ij[23] += R_0_2_2_1 * dm_kl[0];
            vj_ij[58] += R_0_2_2_1 * dm_kl[1];
            vj_ij[24] += R_0_2_3_0 * dm_kl[0];
            vj_ij[59] += R_0_2_3_0 * dm_kl[1];
            vj_ij[25] += R_0_3_0_0 * dm_kl[0];
            vj_ij[60] += R_0_3_0_0 * dm_kl[1];
            vj_ij[26] += R_0_3_0_1 * dm_kl[0];
            vj_ij[61] += R_0_3_0_1 * dm_kl[1];
            vj_ij[27] += R_0_3_0_2 * dm_kl[0];
            vj_ij[62] += R_0_3_0_2 * dm_kl[1];
            vj_ij[28] += R_0_3_1_0 * dm_kl[0];
            vj_ij[63] += R_0_3_1_0 * dm_kl[1];
            vj_ij[29] += R_0_3_1_1 * dm_kl[0];
            vj_ij[64] += R_0_3_1_1 * dm_kl[1];
            vj_ij[30] += R_0_3_2_0 * dm_kl[0];
            vj_ij[65] += R_0_3_2_0 * dm_kl[1];
            vj_ij[31] += R_0_4_0_0 * dm_kl[0];
            vj_ij[66] += R_0_4_0_0 * dm_kl[1];
            vj_ij[32] += R_0_4_0_1 * dm_kl[0];
            vj_ij[67] += R_0_4_0_1 * dm_kl[1];
            vj_ij[33] += R_0_4_1_0 * dm_kl[0];
            vj_ij[68] += R_0_4_1_0 * dm_kl[1];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_ij[34] += R_0_5_0_0 * dm_kl[0];
            vj_ij[69] += R_0_5_0_0 * dm_kl[1];
            }
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 35; ++n) {
                __syncthreads();
                for (int m = 0; m < 2; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+35*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 2; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(2, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 36; n += 16) {
        int kl = n / 9;
        int batch_kl = n - kl * 9;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 144 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(2, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*576+kl*144]);
            }
        }
    }
} }

// TILEX=48, TILEY=12
__global__ static
void md_j_4dm_5_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
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
    double vj_kl[2];
    double dm_kl[2];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 384;
    double *Rp_cache = Rq_cache + 768;
    double *dm_ij_cache = Rp_cache + 64 + tx;
    double *gamma_inc = Rp_cache + 1856 + sq_id;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 832; n += 256) {
        Rq_cache[n] = 0.;
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
            Rq_cache[n+0] = 1e5;
            Rq_cache[n+192] = 1e5;
            Rq_cache[n+384] = 1e5;
            Rq_cache[n+576] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 2) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 384; n += 256) {
        vj_kl_cache[n] = 0.;
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
        int nf3ij_dm = 56 * min(remaining_n_dm, 2);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 56;
            int i = n - i_dm * 56;
            dm_ij_cache[n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[112];
        for (int ij = 0; ij < 112; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 12; ++batch_kl) {
            int task_kl0 = blockIdx.y * 192 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                break;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*12] + q_cond[pair_ij0] < bounds.cutoff) {
                continue;
            }

            int sq_kl = ty + batch_kl * 16;
            int task_ij = task_ij0 + tx;
            int task_kl = task_kl0 + ty;
            double fac = PI_FAC;
            if (task_ij >= npairs_ij || task_kl >= npairs_kl) {
                task_kl = task_kl0;
                fac = 0.;
            }
            int kl_loc0 = pair_kl_loc[task_kl];
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
            double omega = jk.omega;
            boys_fn(gamma_inc, theta, rr, omega, fac, 5, 0, 256);
            if (remaining_n_dm == 1) {
            {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[64];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl[0] += R_0_0_0_5 * dm_ij_cache[80];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[96];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[112];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[128];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[144];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl[0] += R_0_0_1_4 * dm_ij_cache[160];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[176];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[192];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[208];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl[0] += R_0_0_2_3 * dm_ij_cache[224];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[240];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[256];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl[0] += R_0_0_3_2 * dm_ij_cache[272];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[288];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl[0] += R_0_0_4_1 * dm_ij_cache[304];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl[0] += R_0_0_5_0 * dm_ij_cache[320];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[336];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[352];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[368];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[384];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl[0] += R_0_1_0_4 * dm_ij_cache[400];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[416];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[432];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[448];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl[0] += R_0_1_1_3 * dm_ij_cache[464];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[480];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[496];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl[0] += R_0_1_2_2 * dm_ij_cache[512];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[528];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl[0] += R_0_1_3_1 * dm_ij_cache[544];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl[0] += R_0_1_4_0 * dm_ij_cache[560];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[576];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[592];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[608];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl[0] += R_0_2_0_3 * dm_ij_cache[624];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[640];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[656];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl[0] += R_0_2_1_2 * dm_ij_cache[672];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[688];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl[0] += R_0_2_2_1 * dm_ij_cache[704];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl[0] += R_0_2_3_0 * dm_ij_cache[720];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[736];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[752];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl[0] += R_0_3_0_2 * dm_ij_cache[768];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[784];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl[0] += R_0_3_1_1 * dm_ij_cache[800];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl[0] += R_0_3_2_0 * dm_ij_cache[816];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[832];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl[0] += R_0_4_0_1 * dm_ij_cache[848];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl[0] += R_0_4_1_0 * dm_ij_cache[864];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl[0] += R_0_5_0_0 * dm_ij_cache[880];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*192+0] += vj_kl[m];
            } }
            }{
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[4] += R_0_0_0_4 * dm_kl[0];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_ij[5] += R_0_0_0_5 * dm_kl[0];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[6] += R_0_0_1_0 * dm_kl[0];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[7] += R_0_0_1_1 * dm_kl[0];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[8] += R_0_0_1_2 * dm_kl[0];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[9] += R_0_0_1_3 * dm_kl[0];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_ij[10] += R_0_0_1_4 * dm_kl[0];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_ij[11] += R_0_0_2_0 * dm_kl[0];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[12] += R_0_0_2_1 * dm_kl[0];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[13] += R_0_0_2_2 * dm_kl[0];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_ij[14] += R_0_0_2_3 * dm_kl[0];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[15] += R_0_0_3_0 * dm_kl[0];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[16] += R_0_0_3_1 * dm_kl[0];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_ij[17] += R_0_0_3_2 * dm_kl[0];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[18] += R_0_0_4_0 * dm_kl[0];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_ij[19] += R_0_0_4_1 * dm_kl[0];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_ij[20] += R_0_0_5_0 * dm_kl[0];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_ij[21] += R_0_1_0_0 * dm_kl[0];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[22] += R_0_1_0_1 * dm_kl[0];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[23] += R_0_1_0_2 * dm_kl[0];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[24] += R_0_1_0_3 * dm_kl[0];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_ij[25] += R_0_1_0_4 * dm_kl[0];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[26] += R_0_1_1_0 * dm_kl[0];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[27] += R_0_1_1_1 * dm_kl[0];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[28] += R_0_1_1_2 * dm_kl[0];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_ij[29] += R_0_1_1_3 * dm_kl[0];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[30] += R_0_1_2_0 * dm_kl[0];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[31] += R_0_1_2_1 * dm_kl[0];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_ij[32] += R_0_1_2_2 * dm_kl[0];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[33] += R_0_1_3_0 * dm_kl[0];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_ij[34] += R_0_1_3_1 * dm_kl[0];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_ij[35] += R_0_1_4_0 * dm_kl[0];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_ij[36] += R_0_2_0_0 * dm_kl[0];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[37] += R_0_2_0_1 * dm_kl[0];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[38] += R_0_2_0_2 * dm_kl[0];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_ij[39] += R_0_2_0_3 * dm_kl[0];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[40] += R_0_2_1_0 * dm_kl[0];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[41] += R_0_2_1_1 * dm_kl[0];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_ij[42] += R_0_2_1_2 * dm_kl[0];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[43] += R_0_2_2_0 * dm_kl[0];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_ij[44] += R_0_2_2_1 * dm_kl[0];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_ij[45] += R_0_2_3_0 * dm_kl[0];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[46] += R_0_3_0_0 * dm_kl[0];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[47] += R_0_3_0_1 * dm_kl[0];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_ij[48] += R_0_3_0_2 * dm_kl[0];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[49] += R_0_3_1_0 * dm_kl[0];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_ij[50] += R_0_3_1_1 * dm_kl[0];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_ij[51] += R_0_3_2_0 * dm_kl[0];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[52] += R_0_4_0_0 * dm_kl[0];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_ij[53] += R_0_4_0_1 * dm_kl[0];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_ij[54] += R_0_4_1_0 * dm_kl[0];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_ij[55] += R_0_5_0_0 * dm_kl[0];
            }
            } else {
            {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[0*256] * dm_ij_cache[0];
            vj_kl[1] += gamma_inc[0*256] * dm_ij_cache[896];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[912];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[928];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[944];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[64];
            vj_kl[1] += R_0_0_0_4 * dm_ij_cache[960];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_kl[0] += R_0_0_0_5 * dm_ij_cache[80];
            vj_kl[1] += R_0_0_0_5 * dm_ij_cache[976];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[96];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[992];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[112];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[1008];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[128];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[1024];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[144];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[1040];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_kl[0] += R_0_0_1_4 * dm_ij_cache[160];
            vj_kl[1] += R_0_0_1_4 * dm_ij_cache[1056];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[176];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[1072];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[192];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[1088];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[208];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[1104];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_kl[0] += R_0_0_2_3 * dm_ij_cache[224];
            vj_kl[1] += R_0_0_2_3 * dm_ij_cache[1120];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[240];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[1136];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[256];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[1152];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_kl[0] += R_0_0_3_2 * dm_ij_cache[272];
            vj_kl[1] += R_0_0_3_2 * dm_ij_cache[1168];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[288];
            vj_kl[1] += R_0_0_4_0 * dm_ij_cache[1184];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_kl[0] += R_0_0_4_1 * dm_ij_cache[304];
            vj_kl[1] += R_0_0_4_1 * dm_ij_cache[1200];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_kl[0] += R_0_0_5_0 * dm_ij_cache[320];
            vj_kl[1] += R_0_0_5_0 * dm_ij_cache[1216];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[336];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[1232];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[352];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[1248];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[368];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[1264];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[384];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[1280];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_kl[0] += R_0_1_0_4 * dm_ij_cache[400];
            vj_kl[1] += R_0_1_0_4 * dm_ij_cache[1296];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[416];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[1312];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[432];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[1328];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[448];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[1344];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_kl[0] += R_0_1_1_3 * dm_ij_cache[464];
            vj_kl[1] += R_0_1_1_3 * dm_ij_cache[1360];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[480];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[1376];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[496];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[1392];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_kl[0] += R_0_1_2_2 * dm_ij_cache[512];
            vj_kl[1] += R_0_1_2_2 * dm_ij_cache[1408];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[528];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[1424];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_kl[0] += R_0_1_3_1 * dm_ij_cache[544];
            vj_kl[1] += R_0_1_3_1 * dm_ij_cache[1440];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_kl[0] += R_0_1_4_0 * dm_ij_cache[560];
            vj_kl[1] += R_0_1_4_0 * dm_ij_cache[1456];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[576];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[1472];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[592];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[1488];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[608];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[1504];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_kl[0] += R_0_2_0_3 * dm_ij_cache[624];
            vj_kl[1] += R_0_2_0_3 * dm_ij_cache[1520];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[640];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[1536];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[656];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[1552];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_kl[0] += R_0_2_1_2 * dm_ij_cache[672];
            vj_kl[1] += R_0_2_1_2 * dm_ij_cache[1568];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[688];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[1584];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_kl[0] += R_0_2_2_1 * dm_ij_cache[704];
            vj_kl[1] += R_0_2_2_1 * dm_ij_cache[1600];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_kl[0] += R_0_2_3_0 * dm_ij_cache[720];
            vj_kl[1] += R_0_2_3_0 * dm_ij_cache[1616];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[736];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[1632];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[752];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[1648];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_kl[0] += R_0_3_0_2 * dm_ij_cache[768];
            vj_kl[1] += R_0_3_0_2 * dm_ij_cache[1664];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[784];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[1680];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_kl[0] += R_0_3_1_1 * dm_ij_cache[800];
            vj_kl[1] += R_0_3_1_1 * dm_ij_cache[1696];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_kl[0] += R_0_3_2_0 * dm_ij_cache[816];
            vj_kl[1] += R_0_3_2_0 * dm_ij_cache[1712];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[832];
            vj_kl[1] += R_0_4_0_0 * dm_ij_cache[1728];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_kl[0] += R_0_4_0_1 * dm_ij_cache[848];
            vj_kl[1] += R_0_4_0_1 * dm_ij_cache[1744];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_kl[0] += R_0_4_1_0 * dm_ij_cache[864];
            vj_kl[1] += R_0_4_1_0 * dm_ij_cache[1760];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_kl[0] += R_0_5_0_0 * dm_ij_cache[880];
            vj_kl[1] += R_0_5_0_0 * dm_ij_cache[1776];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*192+0] += vj_kl[m];
            } }
            }{
            for (int m = 0; m < 2; ++m) {
                if (m >= remaining_n_dm) break;
                dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0];
            }
            vj_ij[0] += gamma_inc[0*256] * dm_kl[0];
            vj_ij[56] += gamma_inc[0*256] * dm_kl[1];
            double R_0_0_0_1 = zpq * gamma_inc[1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[57] += R_0_0_0_1 * dm_kl[1];
            double R_1_0_0_1 = zpq * gamma_inc[2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[58] += R_0_0_0_2 * dm_kl[1];
            double R_2_0_0_1 = zpq * gamma_inc[3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[59] += R_0_0_0_3 * dm_kl[1];
            double R_3_0_0_1 = zpq * gamma_inc[4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_ij[4] += R_0_0_0_4 * dm_kl[0];
            vj_ij[60] += R_0_0_0_4 * dm_kl[1];
            double R_4_0_0_1 = zpq * gamma_inc[5*256];
            double R_3_0_0_2 = zpq * R_4_0_0_1 + 1 * gamma_inc[4*256];
            double R_2_0_0_3 = zpq * R_3_0_0_2 + 2 * R_3_0_0_1;
            double R_1_0_0_4 = zpq * R_2_0_0_3 + 3 * R_2_0_0_2;
            double R_0_0_0_5 = zpq * R_1_0_0_4 + 4 * R_1_0_0_3;
            vj_ij[5] += R_0_0_0_5 * dm_kl[0];
            vj_ij[61] += R_0_0_0_5 * dm_kl[1];
            double R_0_0_1_0 = ypq * gamma_inc[1*256];
            vj_ij[6] += R_0_0_1_0 * dm_kl[0];
            vj_ij[62] += R_0_0_1_0 * dm_kl[1];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[7] += R_0_0_1_1 * dm_kl[0];
            vj_ij[63] += R_0_0_1_1 * dm_kl[1];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[8] += R_0_0_1_2 * dm_kl[0];
            vj_ij[64] += R_0_0_1_2 * dm_kl[1];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_ij[9] += R_0_0_1_3 * dm_kl[0];
            vj_ij[65] += R_0_0_1_3 * dm_kl[1];
            double R_0_0_1_4 = ypq * R_1_0_0_4;
            vj_ij[10] += R_0_0_1_4 * dm_kl[0];
            vj_ij[66] += R_0_0_1_4 * dm_kl[1];
            double R_1_0_1_0 = ypq * gamma_inc[2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[1*256];
            vj_ij[11] += R_0_0_2_0 * dm_kl[0];
            vj_ij[67] += R_0_0_2_0 * dm_kl[1];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[12] += R_0_0_2_1 * dm_kl[0];
            vj_ij[68] += R_0_0_2_1 * dm_kl[1];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_ij[13] += R_0_0_2_2 * dm_kl[0];
            vj_ij[69] += R_0_0_2_2 * dm_kl[1];
            double R_1_0_1_3 = ypq * R_2_0_0_3;
            double R_0_0_2_3 = ypq * R_1_0_1_3 + 1 * R_1_0_0_3;
            vj_ij[14] += R_0_0_2_3 * dm_kl[0];
            vj_ij[70] += R_0_0_2_3 * dm_kl[1];
            double R_2_0_1_0 = ypq * gamma_inc[3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[15] += R_0_0_3_0 * dm_kl[0];
            vj_ij[71] += R_0_0_3_0 * dm_kl[1];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_ij[16] += R_0_0_3_1 * dm_kl[0];
            vj_ij[72] += R_0_0_3_1 * dm_kl[1];
            double R_2_0_1_2 = ypq * R_3_0_0_2;
            double R_1_0_2_2 = ypq * R_2_0_1_2 + 1 * R_2_0_0_2;
            double R_0_0_3_2 = ypq * R_1_0_2_2 + 2 * R_1_0_1_2;
            vj_ij[17] += R_0_0_3_2 * dm_kl[0];
            vj_ij[73] += R_0_0_3_2 * dm_kl[1];
            double R_3_0_1_0 = ypq * gamma_inc[4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_ij[18] += R_0_0_4_0 * dm_kl[0];
            vj_ij[74] += R_0_0_4_0 * dm_kl[1];
            double R_3_0_1_1 = ypq * R_4_0_0_1;
            double R_2_0_2_1 = ypq * R_3_0_1_1 + 1 * R_3_0_0_1;
            double R_1_0_3_1 = ypq * R_2_0_2_1 + 2 * R_2_0_1_1;
            double R_0_0_4_1 = ypq * R_1_0_3_1 + 3 * R_1_0_2_1;
            vj_ij[19] += R_0_0_4_1 * dm_kl[0];
            vj_ij[75] += R_0_0_4_1 * dm_kl[1];
            double R_4_0_1_0 = ypq * gamma_inc[5*256];
            double R_3_0_2_0 = ypq * R_4_0_1_0 + 1 * gamma_inc[4*256];
            double R_2_0_3_0 = ypq * R_3_0_2_0 + 2 * R_3_0_1_0;
            double R_1_0_4_0 = ypq * R_2_0_3_0 + 3 * R_2_0_2_0;
            double R_0_0_5_0 = ypq * R_1_0_4_0 + 4 * R_1_0_3_0;
            vj_ij[20] += R_0_0_5_0 * dm_kl[0];
            vj_ij[76] += R_0_0_5_0 * dm_kl[1];
            double R_0_1_0_0 = xpq * gamma_inc[1*256];
            vj_ij[21] += R_0_1_0_0 * dm_kl[0];
            vj_ij[77] += R_0_1_0_0 * dm_kl[1];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_ij[22] += R_0_1_0_1 * dm_kl[0];
            vj_ij[78] += R_0_1_0_1 * dm_kl[1];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_ij[23] += R_0_1_0_2 * dm_kl[0];
            vj_ij[79] += R_0_1_0_2 * dm_kl[1];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_ij[24] += R_0_1_0_3 * dm_kl[0];
            vj_ij[80] += R_0_1_0_3 * dm_kl[1];
            double R_0_1_0_4 = xpq * R_1_0_0_4;
            vj_ij[25] += R_0_1_0_4 * dm_kl[0];
            vj_ij[81] += R_0_1_0_4 * dm_kl[1];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_ij[26] += R_0_1_1_0 * dm_kl[0];
            vj_ij[82] += R_0_1_1_0 * dm_kl[1];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_ij[27] += R_0_1_1_1 * dm_kl[0];
            vj_ij[83] += R_0_1_1_1 * dm_kl[1];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_ij[28] += R_0_1_1_2 * dm_kl[0];
            vj_ij[84] += R_0_1_1_2 * dm_kl[1];
            double R_0_1_1_3 = xpq * R_1_0_1_3;
            vj_ij[29] += R_0_1_1_3 * dm_kl[0];
            vj_ij[85] += R_0_1_1_3 * dm_kl[1];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_ij[30] += R_0_1_2_0 * dm_kl[0];
            vj_ij[86] += R_0_1_2_0 * dm_kl[1];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_ij[31] += R_0_1_2_1 * dm_kl[0];
            vj_ij[87] += R_0_1_2_1 * dm_kl[1];
            double R_0_1_2_2 = xpq * R_1_0_2_2;
            vj_ij[32] += R_0_1_2_2 * dm_kl[0];
            vj_ij[88] += R_0_1_2_2 * dm_kl[1];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_ij[33] += R_0_1_3_0 * dm_kl[0];
            vj_ij[89] += R_0_1_3_0 * dm_kl[1];
            double R_0_1_3_1 = xpq * R_1_0_3_1;
            vj_ij[34] += R_0_1_3_1 * dm_kl[0];
            vj_ij[90] += R_0_1_3_1 * dm_kl[1];
            double R_0_1_4_0 = xpq * R_1_0_4_0;
            vj_ij[35] += R_0_1_4_0 * dm_kl[0];
            vj_ij[91] += R_0_1_4_0 * dm_kl[1];
            double R_1_1_0_0 = xpq * gamma_inc[2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[1*256];
            vj_ij[36] += R_0_2_0_0 * dm_kl[0];
            vj_ij[92] += R_0_2_0_0 * dm_kl[1];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[37] += R_0_2_0_1 * dm_kl[0];
            vj_ij[93] += R_0_2_0_1 * dm_kl[1];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_ij[38] += R_0_2_0_2 * dm_kl[0];
            vj_ij[94] += R_0_2_0_2 * dm_kl[1];
            double R_1_1_0_3 = xpq * R_2_0_0_3;
            double R_0_2_0_3 = xpq * R_1_1_0_3 + 1 * R_1_0_0_3;
            vj_ij[39] += R_0_2_0_3 * dm_kl[0];
            vj_ij[95] += R_0_2_0_3 * dm_kl[1];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[40] += R_0_2_1_0 * dm_kl[0];
            vj_ij[96] += R_0_2_1_0 * dm_kl[1];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_ij[41] += R_0_2_1_1 * dm_kl[0];
            vj_ij[97] += R_0_2_1_1 * dm_kl[1];
            double R_1_1_1_2 = xpq * R_2_0_1_2;
            double R_0_2_1_2 = xpq * R_1_1_1_2 + 1 * R_1_0_1_2;
            vj_ij[42] += R_0_2_1_2 * dm_kl[0];
            vj_ij[98] += R_0_2_1_2 * dm_kl[1];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_ij[43] += R_0_2_2_0 * dm_kl[0];
            vj_ij[99] += R_0_2_2_0 * dm_kl[1];
            double R_1_1_2_1 = xpq * R_2_0_2_1;
            double R_0_2_2_1 = xpq * R_1_1_2_1 + 1 * R_1_0_2_1;
            vj_ij[44] += R_0_2_2_1 * dm_kl[0];
            vj_ij[100] += R_0_2_2_1 * dm_kl[1];
            double R_1_1_3_0 = xpq * R_2_0_3_0;
            double R_0_2_3_0 = xpq * R_1_1_3_0 + 1 * R_1_0_3_0;
            vj_ij[45] += R_0_2_3_0 * dm_kl[0];
            vj_ij[101] += R_0_2_3_0 * dm_kl[1];
            double R_2_1_0_0 = xpq * gamma_inc[3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[46] += R_0_3_0_0 * dm_kl[0];
            vj_ij[102] += R_0_3_0_0 * dm_kl[1];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_ij[47] += R_0_3_0_1 * dm_kl[0];
            vj_ij[103] += R_0_3_0_1 * dm_kl[1];
            double R_2_1_0_2 = xpq * R_3_0_0_2;
            double R_1_2_0_2 = xpq * R_2_1_0_2 + 1 * R_2_0_0_2;
            double R_0_3_0_2 = xpq * R_1_2_0_2 + 2 * R_1_1_0_2;
            vj_ij[48] += R_0_3_0_2 * dm_kl[0];
            vj_ij[104] += R_0_3_0_2 * dm_kl[1];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_ij[49] += R_0_3_1_0 * dm_kl[0];
            vj_ij[105] += R_0_3_1_0 * dm_kl[1];
            double R_2_1_1_1 = xpq * R_3_0_1_1;
            double R_1_2_1_1 = xpq * R_2_1_1_1 + 1 * R_2_0_1_1;
            double R_0_3_1_1 = xpq * R_1_2_1_1 + 2 * R_1_1_1_1;
            vj_ij[50] += R_0_3_1_1 * dm_kl[0];
            vj_ij[106] += R_0_3_1_1 * dm_kl[1];
            double R_2_1_2_0 = xpq * R_3_0_2_0;
            double R_1_2_2_0 = xpq * R_2_1_2_0 + 1 * R_2_0_2_0;
            double R_0_3_2_0 = xpq * R_1_2_2_0 + 2 * R_1_1_2_0;
            vj_ij[51] += R_0_3_2_0 * dm_kl[0];
            vj_ij[107] += R_0_3_2_0 * dm_kl[1];
            double R_3_1_0_0 = xpq * gamma_inc[4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[52] += R_0_4_0_0 * dm_kl[0];
            vj_ij[108] += R_0_4_0_0 * dm_kl[1];
            double R_3_1_0_1 = xpq * R_4_0_0_1;
            double R_2_2_0_1 = xpq * R_3_1_0_1 + 1 * R_3_0_0_1;
            double R_1_3_0_1 = xpq * R_2_2_0_1 + 2 * R_2_1_0_1;
            double R_0_4_0_1 = xpq * R_1_3_0_1 + 3 * R_1_2_0_1;
            vj_ij[53] += R_0_4_0_1 * dm_kl[0];
            vj_ij[109] += R_0_4_0_1 * dm_kl[1];
            double R_3_1_1_0 = xpq * R_4_0_1_0;
            double R_2_2_1_0 = xpq * R_3_1_1_0 + 1 * R_3_0_1_0;
            double R_1_3_1_0 = xpq * R_2_2_1_0 + 2 * R_2_1_1_0;
            double R_0_4_1_0 = xpq * R_1_3_1_0 + 3 * R_1_2_1_0;
            vj_ij[54] += R_0_4_1_0 * dm_kl[0];
            vj_ij[110] += R_0_4_1_0 * dm_kl[1];
            double R_4_1_0_0 = xpq * gamma_inc[5*256];
            double R_3_2_0_0 = xpq * R_4_1_0_0 + 1 * gamma_inc[4*256];
            double R_2_3_0_0 = xpq * R_3_2_0_0 + 2 * R_3_1_0_0;
            double R_1_4_0_0 = xpq * R_2_3_0_0 + 3 * R_2_2_0_0;
            double R_0_5_0_0 = xpq * R_1_4_0_0 + 4 * R_1_3_0_0;
            vj_ij[55] += R_0_5_0_0 * dm_kl[0];
            vj_ij[111] += R_0_5_0_0 * dm_kl[1];
            }
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else {
            int task_ij = task_ij0 + tx;
#pragma unroll
            for (int n = 0; n < 56; ++n) {
                __syncthreads();
                for (int m = 0; m < 2; ++m) {
                    vj_cache[thread_id+256*m] = vj_ij[n+56*m];
                }
                for (int stride = 8; stride > 0; stride /= 2) {
                    __syncthreads();
                    if (ty < stride) {
                        for (int m = 0; m < 2; ++m) {
                            vj_cache[thread_id+256*m] += vj_cache[thread_id+256*m + stride*16];
                        }
                    }
                }
                __syncthreads();
                if (ty == 0 && task_ij < npairs_ij) {
                    for (int m = 0; m < min(2, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 12; n += 16) {
        int kl = n / 12;
        int batch_kl = n - kl * 12;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 192 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(2, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*192+kl*192]);
            }
        }
    }
} }

int md_j_4dm_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds,
                      double omega, int dm_size)
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
    int addition_buf = 0;
    if (omega < 0) {
        addition_buf = 256;
    }
    switch (ijkl) {
    case 0: { // lij=0, lkl=0, tilex=21, tiley=21
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 335) / 336, (npairs_kl + 335) / 336, 1);
        cudaFuncSetAttribute(md_j_4dm_0_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6080+addition_buf)*sizeof(double));
        md_j_4dm_0_0<<<blocks, threads, (6080+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 9: { // lij=1, lkl=0, tilex=48, tiley=21
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 335) / 336, 1);
        cudaFuncSetAttribute(md_j_4dm_1_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6080+addition_buf)*sizeof(double));
        md_j_4dm_1_0<<<blocks, threads, (6080+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 10: { // lij=1, lkl=1, tilex=6, tiley=6
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 95) / 96, (npairs_kl + 95) / 96, 1);
        md_j_4dm_1_1<<<blocks, threads, (5568+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 18: { // lij=2, lkl=0, tilex=48, tiley=16
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 255) / 256, 1);
        cudaFuncSetAttribute(md_j_4dm_2_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (5952+addition_buf)*sizeof(double));
        md_j_4dm_2_0<<<blocks, threads, (5952+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 19: { // lij=2, lkl=1, tilex=48, tiley=10
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 159) / 160, 1);
        cudaFuncSetAttribute(md_j_4dm_2_1, cudaFuncAttributeMaxDynamicSharedMemorySize, (5952+addition_buf)*sizeof(double));
        md_j_4dm_2_1<<<blocks, threads, (5952+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 20: { // lij=2, lkl=2, tilex=4, tiley=4
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 63) / 64, (npairs_kl + 63) / 64, 1);
        cudaFuncSetAttribute(md_j_4dm_2_2, cudaFuncAttributeMaxDynamicSharedMemorySize, (6080+addition_buf)*sizeof(double));
        md_j_4dm_2_2<<<blocks, threads, (6080+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 27: { // lij=3, lkl=0, tilex=48, tiley=21
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 335) / 336, 1);
        cudaFuncSetAttribute(md_j_4dm_3_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6080+addition_buf)*sizeof(double));
        md_j_4dm_3_0<<<blocks, threads, (6080+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 28: { // lij=3, lkl=1, tilex=48, tiley=6
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 95) / 96, 1);
        md_j_4dm_3_1<<<blocks, threads, (5824+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 36: { // lij=4, lkl=0, tilex=48, tiley=24
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 383) / 384, 1);
        cudaFuncSetAttribute(md_j_4dm_4_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6048+addition_buf)*sizeof(double));
        md_j_4dm_4_0<<<blocks, threads, (6048+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 37: { // lij=4, lkl=1, tilex=48, tiley=9
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 143) / 144, 1);
        cudaFuncSetAttribute(md_j_4dm_4_1, cudaFuncAttributeMaxDynamicSharedMemorySize, (5984+addition_buf)*sizeof(double));
        md_j_4dm_4_1<<<blocks, threads, (5984+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 45: { // lij=5, lkl=0, tilex=48, tiley=12
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 191) / 192, 1);
        cudaFuncSetAttribute(md_j_4dm_5_0, cudaFuncAttributeMaxDynamicSharedMemorySize, (6080+addition_buf)*sizeof(double));
        md_j_4dm_5_0<<<blocks, threads, (6080+addition_buf)*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    default: return 0;
    }
    return 1;
}
