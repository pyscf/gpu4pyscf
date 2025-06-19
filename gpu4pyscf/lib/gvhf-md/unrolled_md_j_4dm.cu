#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/gamma_inc.cu"
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
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 128;
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
        int nf3ij_dm = 1 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 1;
            int i = n - i_dm * 1;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[8];
        for (int ij = 0; ij < 8; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 21; ++batch_kl) {
            int task_kl0 = blockIdx.y * 336 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*21] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*21] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+336];
            double zkl = Rq_cache[sq_kl+672];
            double akl = Rq_cache[sq_kl+1008];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+16];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[1] += gamma_inc[sq_id+0*256] * dm_kl[1];
            } else {
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+16];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+32];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+48];
            vj_kl[4] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+64];
            vj_kl[5] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+80];
            vj_kl[6] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+96];
            vj_kl[7] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[1] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[2] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[3] += gamma_inc[sq_id+0*256] * dm_kl[3];
            vj_ij[4] += gamma_inc[sq_id+0*256] * dm_kl[4];
            vj_ij[5] += gamma_inc[sq_id+0*256] * dm_kl[5];
            vj_ij[6] += gamma_inc[sq_id+0*256] * dm_kl[6];
            vj_ij[7] += gamma_inc[sq_id+0*256] * dm_kl[7];
            }
        }
        double *vj_cache = Rp_cache;
        if (remaining_n_dm == 1) {
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
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
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 512;
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
        int nf3ij_dm = 4 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 4;
            int i = n - i_dm * 4;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[32];
        for (int ij = 0; ij < 32; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 21; ++batch_kl) {
            int task_kl0 = blockIdx.y * 336 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*21] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+336];
            double zkl = Rq_cache[sq_kl+672];
            double akl = Rq_cache[sq_kl+1008];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+32];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            } else if (remaining_n_dm == 2) {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+64];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+80];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+96];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[5] += R_0_0_0_1 * dm_kl[1];
            vj_ij[2] += R_0_0_1_0 * dm_kl[0];
            vj_ij[6] += R_0_0_1_0 * dm_kl[1];
            vj_ij[3] += R_0_1_0_0 * dm_kl[0];
            vj_ij[7] += R_0_1_0_0 * dm_kl[1];
            } else {
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+64];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+128];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+192];
            vj_kl[4] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+256];
            vj_kl[5] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            vj_kl[6] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+384];
            vj_kl[7] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+448];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+80];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[tx+144];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[tx+208];
            vj_kl[4] += R_0_0_0_1 * dm_ij_cache[tx+272];
            vj_kl[5] += R_0_0_0_1 * dm_ij_cache[tx+336];
            vj_kl[6] += R_0_0_0_1 * dm_ij_cache[tx+400];
            vj_kl[7] += R_0_0_0_1 * dm_ij_cache[tx+464];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+96];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[tx+160];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[tx+224];
            vj_kl[4] += R_0_0_1_0 * dm_ij_cache[tx+288];
            vj_kl[5] += R_0_0_1_0 * dm_ij_cache[tx+352];
            vj_kl[6] += R_0_0_1_0 * dm_ij_cache[tx+416];
            vj_kl[7] += R_0_0_1_0 * dm_ij_cache[tx+480];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+112];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[tx+176];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[tx+240];
            vj_kl[4] += R_0_1_0_0 * dm_ij_cache[tx+304];
            vj_kl[5] += R_0_1_0_0 * dm_ij_cache[tx+368];
            vj_kl[6] += R_0_1_0_0 * dm_ij_cache[tx+432];
            vj_kl[7] += R_0_1_0_0 * dm_ij_cache[tx+496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[8] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[12] += gamma_inc[sq_id+0*256] * dm_kl[3];
            vj_ij[16] += gamma_inc[sq_id+0*256] * dm_kl[4];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[5];
            vj_ij[24] += gamma_inc[sq_id+0*256] * dm_kl[6];
            vj_ij[28] += gamma_inc[sq_id+0*256] * dm_kl[7];
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
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

// TILEX=7, TILEY=7
__global__ static
void md_j_4dm_1_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 112;
    int task_kl0 = blockIdx.y * 112;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+112 <= task_kl0) {
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
    double *Rq_cache = vj_kl_cache + 3584;
    double *Rp_cache = Rq_cache + 448;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 512;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 512; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 112; n += 256) {
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
        } else {
            Rq_cache[n+336] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 8) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 3584; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    for (int batch_ij = 0; batch_ij < 7; ++batch_ij) {
        int task_ij0 = blockIdx.x * 112 + batch_ij * 16;
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
        int nf3ij_dm = 4 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 4;
            int i = n - i_dm * 4;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[32];
        for (int ij = 0; ij < 32; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 7; ++batch_kl) {
            int task_kl0 = blockIdx.y * 112 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*7] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*7] + q_cond[pair_ij0] < bounds.cutoff) {
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
            eval_gamma_inc_fn(gamma_inc, theta_rr, 2, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 2; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+32];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+32];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+112] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+32];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+224] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+32];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+48];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+336] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+64];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+80];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+96];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*448+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+64];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+80];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*448+112] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+64];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+96];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*448+224] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+64];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+112];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*448+336] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[sq_id+0*256] * dm_kl[1];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+64];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+128];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+192];
            vj_kl[4] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+256];
            vj_kl[5] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            vj_kl[6] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+384];
            vj_kl[7] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+448];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+80];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[tx+144];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[tx+208];
            vj_kl[4] += R_0_0_0_1 * dm_ij_cache[tx+272];
            vj_kl[5] += R_0_0_0_1 * dm_ij_cache[tx+336];
            vj_kl[6] += R_0_0_0_1 * dm_ij_cache[tx+400];
            vj_kl[7] += R_0_0_0_1 * dm_ij_cache[tx+464];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+96];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[tx+160];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[tx+224];
            vj_kl[4] += R_0_0_1_0 * dm_ij_cache[tx+288];
            vj_kl[5] += R_0_0_1_0 * dm_ij_cache[tx+352];
            vj_kl[6] += R_0_0_1_0 * dm_ij_cache[tx+416];
            vj_kl[7] += R_0_0_1_0 * dm_ij_cache[tx+480];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+112];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[tx+176];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[tx+240];
            vj_kl[4] += R_0_1_0_0 * dm_ij_cache[tx+304];
            vj_kl[5] += R_0_1_0_0 * dm_ij_cache[tx+368];
            vj_kl[6] += R_0_1_0_0 * dm_ij_cache[tx+432];
            vj_kl[7] += R_0_1_0_0 * dm_ij_cache[tx+496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+64];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[tx+128];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[tx+192];
            vj_kl[4] -= R_0_0_0_1 * dm_ij_cache[tx+256];
            vj_kl[5] -= R_0_0_0_1 * dm_ij_cache[tx+320];
            vj_kl[6] -= R_0_0_0_1 * dm_ij_cache[tx+384];
            vj_kl[7] -= R_0_0_0_1 * dm_ij_cache[tx+448];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+80];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[tx+144];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[tx+208];
            vj_kl[4] -= R_0_0_0_2 * dm_ij_cache[tx+272];
            vj_kl[5] -= R_0_0_0_2 * dm_ij_cache[tx+336];
            vj_kl[6] -= R_0_0_0_2 * dm_ij_cache[tx+400];
            vj_kl[7] -= R_0_0_0_2 * dm_ij_cache[tx+464];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+96];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+160];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+224];
            vj_kl[4] -= R_0_0_1_1 * dm_ij_cache[tx+288];
            vj_kl[5] -= R_0_0_1_1 * dm_ij_cache[tx+352];
            vj_kl[6] -= R_0_0_1_1 * dm_ij_cache[tx+416];
            vj_kl[7] -= R_0_0_1_1 * dm_ij_cache[tx+480];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+112];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+240];
            vj_kl[4] -= R_0_1_0_1 * dm_ij_cache[tx+304];
            vj_kl[5] -= R_0_1_0_1 * dm_ij_cache[tx+368];
            vj_kl[6] -= R_0_1_0_1 * dm_ij_cache[tx+432];
            vj_kl[7] -= R_0_1_0_1 * dm_ij_cache[tx+496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+112] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+64];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[tx+128];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[tx+192];
            vj_kl[4] -= R_0_0_1_0 * dm_ij_cache[tx+256];
            vj_kl[5] -= R_0_0_1_0 * dm_ij_cache[tx+320];
            vj_kl[6] -= R_0_0_1_0 * dm_ij_cache[tx+384];
            vj_kl[7] -= R_0_0_1_0 * dm_ij_cache[tx+448];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+80];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+144];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+208];
            vj_kl[4] -= R_0_0_1_1 * dm_ij_cache[tx+272];
            vj_kl[5] -= R_0_0_1_1 * dm_ij_cache[tx+336];
            vj_kl[6] -= R_0_0_1_1 * dm_ij_cache[tx+400];
            vj_kl[7] -= R_0_0_1_1 * dm_ij_cache[tx+464];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+96];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[tx+160];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[tx+224];
            vj_kl[4] -= R_0_0_2_0 * dm_ij_cache[tx+288];
            vj_kl[5] -= R_0_0_2_0 * dm_ij_cache[tx+352];
            vj_kl[6] -= R_0_0_2_0 * dm_ij_cache[tx+416];
            vj_kl[7] -= R_0_0_2_0 * dm_ij_cache[tx+480];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+112];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+176];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+240];
            vj_kl[4] -= R_0_1_1_0 * dm_ij_cache[tx+304];
            vj_kl[5] -= R_0_1_1_0 * dm_ij_cache[tx+368];
            vj_kl[6] -= R_0_1_1_0 * dm_ij_cache[tx+432];
            vj_kl[7] -= R_0_1_1_0 * dm_ij_cache[tx+496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+224] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+64];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[tx+128];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[tx+192];
            vj_kl[4] -= R_0_1_0_0 * dm_ij_cache[tx+256];
            vj_kl[5] -= R_0_1_0_0 * dm_ij_cache[tx+320];
            vj_kl[6] -= R_0_1_0_0 * dm_ij_cache[tx+384];
            vj_kl[7] -= R_0_1_0_0 * dm_ij_cache[tx+448];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+80];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+144];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+208];
            vj_kl[4] -= R_0_1_0_1 * dm_ij_cache[tx+272];
            vj_kl[5] -= R_0_1_0_1 * dm_ij_cache[tx+336];
            vj_kl[6] -= R_0_1_0_1 * dm_ij_cache[tx+400];
            vj_kl[7] -= R_0_1_0_1 * dm_ij_cache[tx+464];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+224];
            vj_kl[4] -= R_0_1_1_0 * dm_ij_cache[tx+288];
            vj_kl[5] -= R_0_1_1_0 * dm_ij_cache[tx+352];
            vj_kl[6] -= R_0_1_1_0 * dm_ij_cache[tx+416];
            vj_kl[7] -= R_0_1_1_0 * dm_ij_cache[tx+480];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+112];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[tx+176];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[tx+240];
            vj_kl[4] -= R_0_2_0_0 * dm_ij_cache[tx+304];
            vj_kl[5] -= R_0_2_0_0 * dm_ij_cache[tx+368];
            vj_kl[6] -= R_0_2_0_0 * dm_ij_cache[tx+432];
            vj_kl[7] -= R_0_2_0_0 * dm_ij_cache[tx+496];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*448+336] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[4] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[8] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[12] += gamma_inc[sq_id+0*256] * dm_kl[3];
            vj_ij[16] += gamma_inc[sq_id+0*256] * dm_kl[4];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[5];
            vj_ij[24] += gamma_inc[sq_id+0*256] * dm_kl[6];
            vj_ij[28] += gamma_inc[sq_id+0*256] * dm_kl[7];
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
            for (int m = 0; m < 8; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
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
            for (int m = 0; m < 8; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2]; }
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
            for (int m = 0; m < 8; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 28; n += 16) {
        int kl = n / 7;
        int batch_kl = n - kl * 7;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 112 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(8, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*448+kl*112]);
            }
        }
    }
} }

// TILEX=48, TILEY=21
__global__ static
void md_j_4dm_2_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
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
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 1280;
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
        int nf3ij_dm = 10 * min(remaining_n_dm, 8);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 10;
            int i = n - i_dm * 10;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[80];
        for (int ij = 0; ij < 80; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 21; ++batch_kl) {
            int task_kl0 = blockIdx.y * 336 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*21] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+336];
            double zkl = Rq_cache[sq_kl+672];
            double akl = Rq_cache[sq_kl+1008];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+160];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+176];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+192];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+208];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+224];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+240];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+256];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+272];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+288];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[sq_id+0*256] * dm_kl[1];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+160];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+480];
            vj_kl[4] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+640];
            vj_kl[5] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+800];
            vj_kl[6] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+960];
            vj_kl[7] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+1120];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[tx+496];
            vj_kl[4] += R_0_0_0_1 * dm_ij_cache[tx+656];
            vj_kl[5] += R_0_0_0_1 * dm_ij_cache[tx+816];
            vj_kl[6] += R_0_0_0_1 * dm_ij_cache[tx+976];
            vj_kl[7] += R_0_0_0_1 * dm_ij_cache[tx+1136];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[tx+512];
            vj_kl[4] += R_0_0_0_2 * dm_ij_cache[tx+672];
            vj_kl[5] += R_0_0_0_2 * dm_ij_cache[tx+832];
            vj_kl[6] += R_0_0_0_2 * dm_ij_cache[tx+992];
            vj_kl[7] += R_0_0_0_2 * dm_ij_cache[tx+1152];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[tx+528];
            vj_kl[4] += R_0_0_1_0 * dm_ij_cache[tx+688];
            vj_kl[5] += R_0_0_1_0 * dm_ij_cache[tx+848];
            vj_kl[6] += R_0_0_1_0 * dm_ij_cache[tx+1008];
            vj_kl[7] += R_0_0_1_0 * dm_ij_cache[tx+1168];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[tx+544];
            vj_kl[4] += R_0_0_1_1 * dm_ij_cache[tx+704];
            vj_kl[5] += R_0_0_1_1 * dm_ij_cache[tx+864];
            vj_kl[6] += R_0_0_1_1 * dm_ij_cache[tx+1024];
            vj_kl[7] += R_0_0_1_1 * dm_ij_cache[tx+1184];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[tx+560];
            vj_kl[4] += R_0_0_2_0 * dm_ij_cache[tx+720];
            vj_kl[5] += R_0_0_2_0 * dm_ij_cache[tx+880];
            vj_kl[6] += R_0_0_2_0 * dm_ij_cache[tx+1040];
            vj_kl[7] += R_0_0_2_0 * dm_ij_cache[tx+1200];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[tx+576];
            vj_kl[4] += R_0_1_0_0 * dm_ij_cache[tx+736];
            vj_kl[5] += R_0_1_0_0 * dm_ij_cache[tx+896];
            vj_kl[6] += R_0_1_0_0 * dm_ij_cache[tx+1056];
            vj_kl[7] += R_0_1_0_0 * dm_ij_cache[tx+1216];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[tx+592];
            vj_kl[4] += R_0_1_0_1 * dm_ij_cache[tx+752];
            vj_kl[5] += R_0_1_0_1 * dm_ij_cache[tx+912];
            vj_kl[6] += R_0_1_0_1 * dm_ij_cache[tx+1072];
            vj_kl[7] += R_0_1_0_1 * dm_ij_cache[tx+1232];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[tx+608];
            vj_kl[4] += R_0_1_1_0 * dm_ij_cache[tx+768];
            vj_kl[5] += R_0_1_1_0 * dm_ij_cache[tx+928];
            vj_kl[6] += R_0_1_1_0 * dm_ij_cache[tx+1088];
            vj_kl[7] += R_0_1_1_0 * dm_ij_cache[tx+1248];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[tx+624];
            vj_kl[4] += R_0_2_0_0 * dm_ij_cache[tx+784];
            vj_kl[5] += R_0_2_0_0 * dm_ij_cache[tx+944];
            vj_kl[6] += R_0_2_0_0 * dm_ij_cache[tx+1104];
            vj_kl[7] += R_0_2_0_0 * dm_ij_cache[tx+1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 8; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 8; ++m) {
                vj_kl_cache[sq_kl+m*336+0] += vj_kl[m];
            } }
            for (int m = 0; m < 8; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[30] += gamma_inc[sq_id+0*256] * dm_kl[3];
            vj_ij[40] += gamma_inc[sq_id+0*256] * dm_kl[4];
            vj_ij[50] += gamma_inc[sq_id+0*256] * dm_kl[5];
            vj_ij[60] += gamma_inc[sq_id+0*256] * dm_kl[6];
            vj_ij[70] += gamma_inc[sq_id+0*256] * dm_kl[7];
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
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

// TILEX=48, TILEY=13
__global__ static
void md_j_4dm_2_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
    int task_kl0 = blockIdx.y * 208;
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
    double *Rq_cache = vj_kl_cache + 3328;
    double *Rp_cache = Rq_cache + 832;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 640;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 896; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 208; n += 256) {
        int task_kl = blockIdx.y * 208 + n;
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
            Rq_cache[n+208] = ykl;
            Rq_cache[n+416] = zkl;
            Rq_cache[n+624] = akl;
        } else {
            Rq_cache[n+624] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 3328; n += 256) {
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
        int nf3ij_dm = 10 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 10;
            int i = n - i_dm * 10;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[40];
        for (int ij = 0; ij < 40; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 13; ++batch_kl) {
            int task_kl0 = blockIdx.y * 208 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*13] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+208];
            double zkl = Rq_cache[sq_kl+416];
            double akl = Rq_cache[sq_kl+624];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+208] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+64];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+416] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+128];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+624] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+160];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+176];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+192];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+208];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+224];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+240];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+256];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+272];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+288];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*832+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+176];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+208];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+224];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+272];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+288];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*832+208] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+176];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+208];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+224];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+256];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+272];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+288];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*832+416] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*832+624] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[sq_id+0*256] * dm_kl[1];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+160];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+480];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[tx+496];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[tx+512];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[tx+528];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[tx+544];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[tx+560];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[tx+576];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[tx+592];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[tx+608];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+160];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[tx+320];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+176];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[tx+336];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[tx+496];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[tx+192];
            vj_kl[2] -= R_0_0_0_3 * dm_ij_cache[tx+352];
            vj_kl[3] -= R_0_0_0_3 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+208];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+368];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+528];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+224];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[tx+384];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[tx+544];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+240];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[tx+400];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+256];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+416];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+576];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+272];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[tx+432];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[tx+592];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+288];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+448];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+608];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+304];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[tx+464];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+208] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+160];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[tx+320];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+176];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+336];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+496];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+192];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[tx+352];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+208];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[tx+368];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[tx+528];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+224];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[tx+384];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[tx+544];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[tx+240];
            vj_kl[2] -= R_0_0_3_0 * dm_ij_cache[tx+400];
            vj_kl[3] -= R_0_0_3_0 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+256];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+416];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+576];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+272];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+432];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+592];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+288];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[tx+448];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[tx+608];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+304];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[tx+464];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+416] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[tx+320];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+336];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+496];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[tx+352];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+368];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+528];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+384];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+544];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[tx+400];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[tx+416];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[tx+576];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[tx+432];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[tx+592];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+288];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[tx+448];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[tx+608];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[tx+304];
            vj_kl[2] -= R_0_3_0_0 * dm_ij_cache[tx+464];
            vj_kl[3] -= R_0_3_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*832+624] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[30] += gamma_inc[sq_id+0*256] * dm_kl[3];
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 52; n += 16) {
        int kl = n / 13;
        int batch_kl = n - kl * 13;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 208 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*832+kl*208]);
            }
        }
    }
} }

// TILEX=5, TILEY=5
__global__ static
void md_j_4dm_2_2(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 80;
    int task_kl0 = blockIdx.y * 80;
    int pair_ij0 = pair_ij_mapping[task_ij0];
    int pair_kl0 = pair_kl_mapping[task_kl0];
    float *q_cond = bounds.q_cond;
    if (q_cond[pair_ij0] + q_cond[pair_kl0] < bounds.cutoff) {
        return;
    }
    if (pair_ij_mapping == pair_kl_mapping && task_ij0+80 <= task_kl0) {
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
    double *Rq_cache = vj_kl_cache + 3200;
    double *Rp_cache = Rq_cache + 320;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 640;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 384; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 80; n += 256) {
        int task_kl = blockIdx.y * 80 + n;
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
            Rq_cache[n+80] = ykl;
            Rq_cache[n+160] = zkl;
            Rq_cache[n+240] = akl;
        } else {
            Rq_cache[n+240] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 3200; n += 256) {
        vj_kl_cache[n] = 0.;
    }
    for (int batch_ij = 0; batch_ij < 5; ++batch_ij) {
        int task_ij0 = blockIdx.x * 80 + batch_ij * 16;
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
        int nf3ij_dm = 10 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 10;
            int i = n - i_dm * 10;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[40];
        for (int ij = 0; ij < 40; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 5; ++batch_kl) {
            int task_kl0 = blockIdx.y * 80 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            if (pair_ij_mapping == pair_kl_mapping && task_ij0+16 <= task_kl0) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*5] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*5] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+80];
            double zkl = Rq_cache[sq_kl+160];
            double akl = Rq_cache[sq_kl+240];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+48];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+64];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+96];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+112];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+128];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+80] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+0];
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+16];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[tx+32];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+48];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[tx+64];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+80];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+96];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[tx+112];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+128];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+160] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+64];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+112];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+128];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+240] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+0];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+16];
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[tx+32];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+48];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+64];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[tx+80];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+96];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+112];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+128];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+320] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+0];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+16];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+32];
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+48];
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[tx+64];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[tx+80];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+96];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+112];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[tx+128];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+400] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+128];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+480] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+0];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+16];
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[tx+32];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+48];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+64];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+80];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+96];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+112];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+128];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+560] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+0];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+16];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+32];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+48];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+64];
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[tx+80];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+96];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+112];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+128];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+640] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+0];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+16];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+32];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+48];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+64];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+80];
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+96];
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[tx+112];
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[tx+128];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[tx+144];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+720] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+160];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+176];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+192];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+208];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+224];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+240];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+256];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+272];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+288];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+176];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+208];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+224];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+256];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+272];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+288];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+80] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+160];
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[tx+176];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_4 * dm_ij_cache[tx+192];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+208];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[tx+224];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[tx+240];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+256];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[tx+272];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+288];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+160] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+176];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+208];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+224];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+256];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+272];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+288];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+240] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+160];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+176];
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[tx+192];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+208];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[tx+224];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[tx+240];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+256];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+272];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+288];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+320] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+160];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+176];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[tx+192];
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[tx+208];
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[tx+224];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_4_0 * dm_ij_cache[tx+240];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+256];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+272];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[tx+288];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+400] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+480] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+176];
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+208];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+224];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+256];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[tx+272];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+560] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+224];
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+272];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[tx+288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+640] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[tx+288];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_4_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*800+720] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[sq_id+0*256] * dm_kl[1];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+160];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+480];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[tx+496];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[tx+512];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[tx+528];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[tx+544];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[tx+560];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[tx+576];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[tx+592];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[tx+608];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+160];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[tx+320];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+176];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[tx+336];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[tx+496];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[tx+192];
            vj_kl[2] -= R_0_0_0_3 * dm_ij_cache[tx+352];
            vj_kl[3] -= R_0_0_0_3 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+208];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+368];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+528];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+224];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[tx+384];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[tx+544];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+240];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[tx+400];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+256];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+416];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+576];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+272];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[tx+432];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[tx+592];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+288];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+448];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+608];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+304];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[tx+464];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+80] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+160];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[tx+320];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[tx+480];
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_0_0_3 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_0_0_3 * dm_ij_cache[tx+496];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] += R_0_0_0_4 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_4 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_0_0_4 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_0_0_4 * dm_ij_cache[tx+512];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[tx+528];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_0_1_3 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_0_1_3 * dm_ij_cache[tx+544];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_0_2_2 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_0_2_2 * dm_ij_cache[tx+560];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[tx+576];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_1_0_3 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_1_0_3 * dm_ij_cache[tx+592];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[tx+608];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_2_0_2 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_2_0_2 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+160] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+160];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[tx+320];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+176];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+336];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+496];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+192];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[tx+352];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+208];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[tx+368];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[tx+528];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+224];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[tx+384];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[tx+544];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[tx+240];
            vj_kl[2] -= R_0_0_3_0 * dm_ij_cache[tx+400];
            vj_kl[3] -= R_0_0_3_0 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+256];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+416];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+576];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+272];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+432];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+592];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+288];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[tx+448];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[tx+608];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+304];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[tx+464];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+240] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+160];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[tx+320];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[tx+480];
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[tx+496];
            vj_kl[0] += R_0_0_1_3 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_1_3 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_0_1_3 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_0_1_3 * dm_ij_cache[tx+512];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[tx+528];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_0_2_2 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_0_2_2 * dm_ij_cache[tx+544];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_0_3_1 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_0_3_1 * dm_ij_cache[tx+560];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[tx+576];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[tx+592];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[tx+608];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+320] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+160];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[tx+320];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[tx+480];
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[tx+496];
            vj_kl[0] += R_0_0_2_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_2_2 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_0_2_2 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_0_2_2 * dm_ij_cache[tx+512];
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_0_3_0 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_0_3_0 * dm_ij_cache[tx+528];
            vj_kl[0] += R_0_0_3_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_3_1 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_0_3_1 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_0_3_1 * dm_ij_cache[tx+544];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] += R_0_0_4_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_4_0 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_0_4_0 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_0_4_0 * dm_ij_cache[tx+560];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[tx+576];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[tx+592];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_1_3_0 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_1_3_0 * dm_ij_cache[tx+608];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_2_2_0 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_2_2_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+400] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[tx+320];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+336];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+496];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[tx+352];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+368];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+528];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+384];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+544];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[tx+400];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[tx+416];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[tx+576];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[tx+432];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[tx+592];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+288];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[tx+448];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[tx+608];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[tx+304];
            vj_kl[2] -= R_0_3_0_0 * dm_ij_cache[tx+464];
            vj_kl[3] -= R_0_3_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+480] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[tx+320];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[tx+480];
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[tx+496];
            vj_kl[0] += R_0_1_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_1_0_3 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_1_0_3 * dm_ij_cache[tx+512];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[tx+528];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[tx+544];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[tx+560];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[tx+576];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_2_0_2 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_2_0_2 * dm_ij_cache[tx+592];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[tx+608];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_3_0_1 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_3_0_1 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+560] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[tx+320];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[tx+480];
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[tx+496];
            vj_kl[0] += R_0_1_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_1_1_2 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_1_1_2 * dm_ij_cache[tx+512];
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[tx+528];
            vj_kl[0] += R_0_1_2_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_1_2_1 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_1_2_1 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_1_2_1 * dm_ij_cache[tx+544];
            vj_kl[0] += R_0_1_3_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_1_3_0 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_1_3_0 * dm_ij_cache[tx+560];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[tx+576];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[tx+592];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_2_2_0 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_2_2_0 * dm_ij_cache[tx+608];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_3_1_0 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_3_1_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+640] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[tx+320];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[tx+480];
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[tx+336];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[tx+496];
            vj_kl[0] += R_0_2_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl[2] += R_0_2_0_2 * dm_ij_cache[tx+352];
            vj_kl[3] += R_0_2_0_2 * dm_ij_cache[tx+512];
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[tx+368];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[tx+528];
            vj_kl[0] += R_0_2_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl[2] += R_0_2_1_1 * dm_ij_cache[tx+384];
            vj_kl[3] += R_0_2_1_1 * dm_ij_cache[tx+544];
            vj_kl[0] += R_0_2_2_0 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl[2] += R_0_2_2_0 * dm_ij_cache[tx+400];
            vj_kl[3] += R_0_2_2_0 * dm_ij_cache[tx+560];
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl[2] += R_0_3_0_0 * dm_ij_cache[tx+416];
            vj_kl[3] += R_0_3_0_0 * dm_ij_cache[tx+576];
            vj_kl[0] += R_0_3_0_1 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl[2] += R_0_3_0_1 * dm_ij_cache[tx+432];
            vj_kl[3] += R_0_3_0_1 * dm_ij_cache[tx+592];
            vj_kl[0] += R_0_3_1_0 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_3_1_0 * dm_ij_cache[tx+288];
            vj_kl[2] += R_0_3_1_0 * dm_ij_cache[tx+448];
            vj_kl[3] += R_0_3_1_0 * dm_ij_cache[tx+608];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] += R_0_4_0_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_4_0_0 * dm_ij_cache[tx+304];
            vj_kl[2] += R_0_4_0_0 * dm_ij_cache[tx+464];
            vj_kl[3] += R_0_4_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*800+720] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[10] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[30] += gamma_inc[sq_id+0*256] * dm_kl[3];
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+2]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+4]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+5]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+6]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+7]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+8]; }
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+9]; }
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 50; n += 16) {
        int kl = n / 5;
        int batch_kl = n - kl * 5;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 80 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*800+kl*80]);
            }
        }
    }
} }

// TILEX=48, TILEY=29
__global__ static
void md_j_4dm_3_0(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
{
    int *pair_ij_mapping = bounds.pair_ij_mapping;
    int *pair_kl_mapping = bounds.pair_kl_mapping;
    int task_ij0 = blockIdx.x * 768;
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
    double *Rq_cache = vj_kl_cache + 1856;
    double *Rp_cache = Rq_cache + 1856;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 1280;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 1920; n += 256) {
        Rq_cache[n] = 0.;
    }
    __syncthreads();

    for (int n = thread_id; n < 464; n += 256) {
        int task_kl = blockIdx.y * 464 + n;
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
            Rq_cache[n+464] = ykl;
            Rq_cache[n+928] = zkl;
            Rq_cache[n+1392] = akl;
        } else {
            Rq_cache[n+1392] = 1.;
        }
    }

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 1856; n += 256) {
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
        int nf3ij_dm = 20 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 20;
            int i = n - i_dm * 20;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[80];
        for (int ij = 0; ij < 80; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 29; ++batch_kl) {
            int task_kl0 = blockIdx.y * 464 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
                qd_kl_max[batch_kl+blockIdx.y*29] + q_cond[pair_ij0] < bounds.cutoff) {
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
            double ykl = Rq_cache[sq_kl+464];
            double zkl = Rq_cache[sq_kl+928];
            double akl = Rq_cache[sq_kl+1392];
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
            if (remaining_n_dm == 1) {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+96];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+128];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+144];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+240];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*464+0] += vj_kl[m];
            } }
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+336];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+352];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[tx+368];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+384];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+400];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+416];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+432];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+448];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[tx+464];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+480];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+496];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+512];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+528];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+544];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+560];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+576];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+592];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+608];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*464+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[1];
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
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+640];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+960];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+336];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[tx+656];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[tx+976];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+352];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[tx+672];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[tx+992];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[tx+368];
            vj_kl[2] += R_0_0_0_3 * dm_ij_cache[tx+688];
            vj_kl[3] += R_0_0_0_3 * dm_ij_cache[tx+1008];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+384];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[tx+704];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[tx+1024];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+400];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[tx+720];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[tx+1040];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+416];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[tx+736];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[tx+1056];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+432];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[tx+752];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[tx+1072];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+448];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[tx+768];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[tx+1088];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[tx+464];
            vj_kl[2] += R_0_0_3_0 * dm_ij_cache[tx+784];
            vj_kl[3] += R_0_0_3_0 * dm_ij_cache[tx+1104];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+480];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[tx+800];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[tx+1120];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+496];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[tx+816];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[tx+1136];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+512];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[tx+832];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[tx+1152];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+528];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[tx+848];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[tx+1168];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+544];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[tx+864];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[tx+1184];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+560];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[tx+880];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[tx+1200];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+576];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[tx+896];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[tx+1216];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+592];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[tx+912];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[tx+1232];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+608];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[tx+928];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[tx+1248];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[tx+624];
            vj_kl[2] += R_0_3_0_0 * dm_ij_cache[tx+944];
            vj_kl[3] += R_0_3_0_0 * dm_ij_cache[tx+1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*464+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[40] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[60] += gamma_inc[sq_id+0*256] * dm_kl[3];
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
            }
        }
    }
    for (int n = tx; n < 29; n += 16) {
        int kl = n / 29;
        int batch_kl = n - kl * 29;
        int sq_kl = ty + batch_kl * 16;
        int task_kl = blockIdx.y * 464 + sq_kl;
        if (task_kl < npairs_kl) {
            int kl_loc0 = pair_kl_loc[task_kl];
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*464+kl*464]);
            }
        }
    }
} }

// TILEX=48, TILEY=11
__global__ static
void md_j_4dm_3_1(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds, int dm_size)
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
    double vj_kl[4];
    double dm_kl[4];
    unsigned int lane_id = sq_id % 32;
    unsigned int group_id = lane_id / 16;
    unsigned int mask = 0xffff << (group_id * 16);

    int npairs_ij = bounds.npairs_ij;
    int npairs_kl = bounds.npairs_kl;
    extern __shared__ double vj_kl_cache[];
    double *Rq_cache = vj_kl_cache + 2816;
    double *Rp_cache = Rq_cache + 704;
    double *dm_ij_cache = Rp_cache + 64;
    double *gamma_inc = dm_ij_cache + 1280;
    float *qd_ij_max = bounds.qd_ij_max;
    float *qd_kl_max = bounds.qd_kl_max;

    for (int n = thread_id; n < 768; n += 256) {
        Rq_cache[n] = 0.;
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

for (int dm_offset = 0; dm_offset < jk.n_dm; dm_offset += 4) {
    int remaining_n_dm = jk.n_dm - dm_offset;
    double *dm = jk.dm + dm_offset * dm_size;
    double *vj = jk.vj + dm_offset * dm_size;
    __syncthreads();
    for (int n = thread_id; n < 2816; n += 256) {
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
        int nf3ij_dm = 20 * min(remaining_n_dm, 4);
        for (int n = ty; n < nf3ij_dm; n += 16) {
            int i_dm = n / 20;
            int i = n - i_dm * 20;
            dm_ij_cache[tx+n*16] = dm[i_dm*dm_size+ij_loc0+i];
        }
        double vj_ij[80];
        for (int ij = 0; ij < 80; ++ij) {
            vj_ij[ij] = 0;
        }
        for (int batch_kl = 0; batch_kl < 11; ++batch_kl) {
            int task_kl0 = blockIdx.y * 176 + batch_kl * 16;
            if (task_kl0 >= npairs_kl) {
                continue;
            }
            int pair_ij0 = pair_ij_mapping[task_ij0];
            int pair_kl0 = pair_kl_mapping[task_kl0];
            if (qd_ij_max[batch_ij+blockIdx.x*48] + q_cond[pair_kl0] < bounds.cutoff &&
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
            eval_gamma_inc_fn(gamma_inc, theta_rr, 4, sq_id, 256);
            double a2 = -2. * theta;
            gamma_inc[sq_id] *= fac;
            for (int i = 1; i <= 4; i++) {
                fac *= a2;
                gamma_inc[sq_id+i*256] *= fac;
            }
            if (remaining_n_dm == 1) {
            {
            vj_kl[0] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+48];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+64];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+80];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+96];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+112];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+128];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+144];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+160];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+176];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+192];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+208];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+224];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+240];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+256];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+272];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+288];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+0] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[tx+48];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+80];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[tx+96];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+112];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[tx+128];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[tx+144];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+176];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+208];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+224];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+256];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[tx+272];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+288];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+176] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[tx+48];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+64];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[tx+96];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+112];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[tx+128];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[tx+144];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+224];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+272];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[tx+288];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+352] += vj_kl[m];
            } }
            vj_kl[0] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[tx+48];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+64];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+80];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+96];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+112];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[tx+144];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[tx+288];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[tx+304];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+528] += vj_kl[m];
            } }
            }{
            dm_kl[0] = 1 * dm[kl_loc0+0];
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
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
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[16] += R_0_2_0_0 * dm_kl[0];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_ij[17] += R_0_2_0_1 * dm_kl[0];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_ij[18] += R_0_2_1_0 * dm_kl[0];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            dm_kl[0] = -1 * dm[kl_loc0+1];
            vj_ij[0] += R_0_0_0_1 * dm_kl[0];
            vj_ij[1] += R_0_0_0_2 * dm_kl[0];
            vj_ij[2] += R_0_0_0_3 * dm_kl[0];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
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
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
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
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[19] += R_0_4_0_0 * dm_kl[0];
            }
            } else if (remaining_n_dm == 2) {
            {
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+336];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+352];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[tx+368];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+384];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+400];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+416];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+432];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+448];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[tx+464];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+480];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+496];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+512];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+528];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+544];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+560];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+576];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+592];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+608];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*704+0] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+320];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+336];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[tx+352];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_0_4 * dm_ij_cache[tx+368];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+384];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+400];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[tx+416];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+432];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[tx+448];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[tx+464];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+176];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+496];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+208];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+528];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+224];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[tx+544];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+256];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+576];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[tx+272];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[tx+592];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+288];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[tx+608];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[tx+304];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*704+176] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+320];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+336];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+352];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[tx+368];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+384];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+400];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[tx+416];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[tx+432];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[tx+448];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_0_4_0 * dm_ij_cache[tx+464];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+496];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+528];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+224];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[tx+544];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+576];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+272];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[tx+592];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[tx+288];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[tx+608];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[tx+304];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*704+352] += vj_kl[m];
            } }
            for (int m = 0; m < 2; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+320];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+336];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+352];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[tx+368];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+384];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+400];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[tx+416];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+432];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[tx+448];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[tx+464];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+480];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+496];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[tx+512];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+528];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[tx+544];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[tx+560];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[tx+576];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[tx+592];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[tx+288];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[tx+608];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[tx+304];
            vj_kl[1] -= R_0_4_0_0 * dm_ij_cache[tx+624];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 2; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 2; ++m) {
                vj_kl_cache[sq_kl+m*704+528] += vj_kl[m];
            } }
            }{
            for (int m = 0; m < 2; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[1];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[21] += R_0_0_0_1 * dm_kl[1];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[22] += R_0_0_0_2 * dm_kl[1];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[23] += R_0_0_0_3 * dm_kl[1];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_ij[4] += R_0_0_1_0 * dm_kl[0];
            vj_ij[24] += R_0_0_1_0 * dm_kl[1];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_ij[5] += R_0_0_1_1 * dm_kl[0];
            vj_ij[25] += R_0_0_1_1 * dm_kl[1];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_ij[6] += R_0_0_1_2 * dm_kl[0];
            vj_ij[26] += R_0_0_1_2 * dm_kl[1];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[7] += R_0_0_2_0 * dm_kl[0];
            vj_ij[27] += R_0_0_2_0 * dm_kl[1];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_ij[8] += R_0_0_2_1 * dm_kl[0];
            vj_ij[28] += R_0_0_2_1 * dm_kl[1];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[29] += R_0_0_3_0 * dm_kl[1];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
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
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
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
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
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
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
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
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
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
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_ij[19] += R_0_4_0_0 * dm_kl[0];
            vj_ij[39] += R_0_4_0_0 * dm_kl[1];
            }
            } else {
            {
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+0];
            vj_kl[1] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+320];
            vj_kl[2] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+640];
            vj_kl[3] += gamma_inc[sq_id+0*256] * dm_ij_cache[tx+960];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] += R_0_0_0_1 * dm_ij_cache[tx+336];
            vj_kl[2] += R_0_0_0_1 * dm_ij_cache[tx+656];
            vj_kl[3] += R_0_0_0_1 * dm_ij_cache[tx+976];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] += R_0_0_0_2 * dm_ij_cache[tx+352];
            vj_kl[2] += R_0_0_0_2 * dm_ij_cache[tx+672];
            vj_kl[3] += R_0_0_0_2 * dm_ij_cache[tx+992];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_kl[0] += R_0_0_0_3 * dm_ij_cache[tx+48];
            vj_kl[1] += R_0_0_0_3 * dm_ij_cache[tx+368];
            vj_kl[2] += R_0_0_0_3 * dm_ij_cache[tx+688];
            vj_kl[3] += R_0_0_0_3 * dm_ij_cache[tx+1008];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_1_0 * dm_ij_cache[tx+64];
            vj_kl[1] += R_0_0_1_0 * dm_ij_cache[tx+384];
            vj_kl[2] += R_0_0_1_0 * dm_ij_cache[tx+704];
            vj_kl[3] += R_0_0_1_0 * dm_ij_cache[tx+1024];
            double R_0_0_1_1 = ypq * R_1_0_0_1;
            vj_kl[0] += R_0_0_1_1 * dm_ij_cache[tx+80];
            vj_kl[1] += R_0_0_1_1 * dm_ij_cache[tx+400];
            vj_kl[2] += R_0_0_1_1 * dm_ij_cache[tx+720];
            vj_kl[3] += R_0_0_1_1 * dm_ij_cache[tx+1040];
            double R_0_0_1_2 = ypq * R_1_0_0_2;
            vj_kl[0] += R_0_0_1_2 * dm_ij_cache[tx+96];
            vj_kl[1] += R_0_0_1_2 * dm_ij_cache[tx+416];
            vj_kl[2] += R_0_0_1_2 * dm_ij_cache[tx+736];
            vj_kl[3] += R_0_0_1_2 * dm_ij_cache[tx+1056];
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_0_2_0 * dm_ij_cache[tx+112];
            vj_kl[1] += R_0_0_2_0 * dm_ij_cache[tx+432];
            vj_kl[2] += R_0_0_2_0 * dm_ij_cache[tx+752];
            vj_kl[3] += R_0_0_2_0 * dm_ij_cache[tx+1072];
            double R_1_0_1_1 = ypq * R_2_0_0_1;
            double R_0_0_2_1 = ypq * R_1_0_1_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_0_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] += R_0_0_2_1 * dm_ij_cache[tx+448];
            vj_kl[2] += R_0_0_2_1 * dm_ij_cache[tx+768];
            vj_kl[3] += R_0_0_2_1 * dm_ij_cache[tx+1088];
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_kl[0] += R_0_0_3_0 * dm_ij_cache[tx+144];
            vj_kl[1] += R_0_0_3_0 * dm_ij_cache[tx+464];
            vj_kl[2] += R_0_0_3_0 * dm_ij_cache[tx+784];
            vj_kl[3] += R_0_0_3_0 * dm_ij_cache[tx+1104];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_1_0_0 * dm_ij_cache[tx+160];
            vj_kl[1] += R_0_1_0_0 * dm_ij_cache[tx+480];
            vj_kl[2] += R_0_1_0_0 * dm_ij_cache[tx+800];
            vj_kl[3] += R_0_1_0_0 * dm_ij_cache[tx+1120];
            double R_0_1_0_1 = xpq * R_1_0_0_1;
            vj_kl[0] += R_0_1_0_1 * dm_ij_cache[tx+176];
            vj_kl[1] += R_0_1_0_1 * dm_ij_cache[tx+496];
            vj_kl[2] += R_0_1_0_1 * dm_ij_cache[tx+816];
            vj_kl[3] += R_0_1_0_1 * dm_ij_cache[tx+1136];
            double R_0_1_0_2 = xpq * R_1_0_0_2;
            vj_kl[0] += R_0_1_0_2 * dm_ij_cache[tx+192];
            vj_kl[1] += R_0_1_0_2 * dm_ij_cache[tx+512];
            vj_kl[2] += R_0_1_0_2 * dm_ij_cache[tx+832];
            vj_kl[3] += R_0_1_0_2 * dm_ij_cache[tx+1152];
            double R_0_1_1_0 = xpq * R_1_0_1_0;
            vj_kl[0] += R_0_1_1_0 * dm_ij_cache[tx+208];
            vj_kl[1] += R_0_1_1_0 * dm_ij_cache[tx+528];
            vj_kl[2] += R_0_1_1_0 * dm_ij_cache[tx+848];
            vj_kl[3] += R_0_1_1_0 * dm_ij_cache[tx+1168];
            double R_0_1_1_1 = xpq * R_1_0_1_1;
            vj_kl[0] += R_0_1_1_1 * dm_ij_cache[tx+224];
            vj_kl[1] += R_0_1_1_1 * dm_ij_cache[tx+544];
            vj_kl[2] += R_0_1_1_1 * dm_ij_cache[tx+864];
            vj_kl[3] += R_0_1_1_1 * dm_ij_cache[tx+1184];
            double R_0_1_2_0 = xpq * R_1_0_2_0;
            vj_kl[0] += R_0_1_2_0 * dm_ij_cache[tx+240];
            vj_kl[1] += R_0_1_2_0 * dm_ij_cache[tx+560];
            vj_kl[2] += R_0_1_2_0 * dm_ij_cache[tx+880];
            vj_kl[3] += R_0_1_2_0 * dm_ij_cache[tx+1200];
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
            vj_kl[0] += R_0_2_0_0 * dm_ij_cache[tx+256];
            vj_kl[1] += R_0_2_0_0 * dm_ij_cache[tx+576];
            vj_kl[2] += R_0_2_0_0 * dm_ij_cache[tx+896];
            vj_kl[3] += R_0_2_0_0 * dm_ij_cache[tx+1216];
            double R_1_1_0_1 = xpq * R_2_0_0_1;
            double R_0_2_0_1 = xpq * R_1_1_0_1 + 1 * R_1_0_0_1;
            vj_kl[0] += R_0_2_0_1 * dm_ij_cache[tx+272];
            vj_kl[1] += R_0_2_0_1 * dm_ij_cache[tx+592];
            vj_kl[2] += R_0_2_0_1 * dm_ij_cache[tx+912];
            vj_kl[3] += R_0_2_0_1 * dm_ij_cache[tx+1232];
            double R_1_1_1_0 = xpq * R_2_0_1_0;
            double R_0_2_1_0 = xpq * R_1_1_1_0 + 1 * R_1_0_1_0;
            vj_kl[0] += R_0_2_1_0 * dm_ij_cache[tx+288];
            vj_kl[1] += R_0_2_1_0 * dm_ij_cache[tx+608];
            vj_kl[2] += R_0_2_1_0 * dm_ij_cache[tx+928];
            vj_kl[3] += R_0_2_1_0 * dm_ij_cache[tx+1248];
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_kl[0] += R_0_3_0_0 * dm_ij_cache[tx+304];
            vj_kl[1] += R_0_3_0_0 * dm_ij_cache[tx+624];
            vj_kl[2] += R_0_3_0_0 * dm_ij_cache[tx+944];
            vj_kl[3] += R_0_3_0_0 * dm_ij_cache[tx+1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+0] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_0_1 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_0_1 * dm_ij_cache[tx+320];
            vj_kl[2] -= R_0_0_0_1 * dm_ij_cache[tx+640];
            vj_kl[3] -= R_0_0_0_1 * dm_ij_cache[tx+960];
            vj_kl[0] -= R_0_0_0_2 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_0_2 * dm_ij_cache[tx+336];
            vj_kl[2] -= R_0_0_0_2 * dm_ij_cache[tx+656];
            vj_kl[3] -= R_0_0_0_2 * dm_ij_cache[tx+976];
            vj_kl[0] -= R_0_0_0_3 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_0_3 * dm_ij_cache[tx+352];
            vj_kl[2] -= R_0_0_0_3 * dm_ij_cache[tx+672];
            vj_kl[3] -= R_0_0_0_3 * dm_ij_cache[tx+992];
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_0_3 = zpq * R_2_0_0_2 + 2 * R_2_0_0_1;
            double R_0_0_0_4 = zpq * R_1_0_0_3 + 3 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_0_4 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_0_4 * dm_ij_cache[tx+368];
            vj_kl[2] -= R_0_0_0_4 * dm_ij_cache[tx+688];
            vj_kl[3] -= R_0_0_0_4 * dm_ij_cache[tx+1008];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+384];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+704];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+1024];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+400];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[tx+720];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[tx+1040];
            double R_0_0_1_3 = ypq * R_1_0_0_3;
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[tx+416];
            vj_kl[2] -= R_0_0_1_3 * dm_ij_cache[tx+736];
            vj_kl[3] -= R_0_0_1_3 * dm_ij_cache[tx+1056];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+432];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[tx+752];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[tx+1072];
            double R_1_0_1_2 = ypq * R_2_0_0_2;
            double R_0_0_2_2 = ypq * R_1_0_1_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[tx+448];
            vj_kl[2] -= R_0_0_2_2 * dm_ij_cache[tx+768];
            vj_kl[3] -= R_0_0_2_2 * dm_ij_cache[tx+1088];
            double R_2_0_1_1 = ypq * R_3_0_0_1;
            double R_1_0_2_1 = ypq * R_2_0_1_1 + 1 * R_2_0_0_1;
            double R_0_0_3_1 = ypq * R_1_0_2_1 + 2 * R_1_0_1_1;
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[tx+464];
            vj_kl[2] -= R_0_0_3_1 * dm_ij_cache[tx+784];
            vj_kl[3] -= R_0_0_3_1 * dm_ij_cache[tx+1104];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+160];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+480];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+800];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+1120];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+176];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+496];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[tx+816];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[tx+1136];
            double R_0_1_0_3 = xpq * R_1_0_0_3;
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[tx+192];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[tx+512];
            vj_kl[2] -= R_0_1_0_3 * dm_ij_cache[tx+832];
            vj_kl[3] -= R_0_1_0_3 * dm_ij_cache[tx+1152];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+208];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+528];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+848];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+1168];
            double R_0_1_1_2 = xpq * R_1_0_1_2;
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+224];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[tx+544];
            vj_kl[2] -= R_0_1_1_2 * dm_ij_cache[tx+864];
            vj_kl[3] -= R_0_1_1_2 * dm_ij_cache[tx+1184];
            double R_0_1_2_1 = xpq * R_1_0_2_1;
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+240];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[tx+560];
            vj_kl[2] -= R_0_1_2_1 * dm_ij_cache[tx+880];
            vj_kl[3] -= R_0_1_2_1 * dm_ij_cache[tx+1200];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+256];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+576];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[tx+896];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[tx+1216];
            double R_1_1_0_2 = xpq * R_2_0_0_2;
            double R_0_2_0_2 = xpq * R_1_1_0_2 + 1 * R_1_0_0_2;
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[tx+272];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[tx+592];
            vj_kl[2] -= R_0_2_0_2 * dm_ij_cache[tx+912];
            vj_kl[3] -= R_0_2_0_2 * dm_ij_cache[tx+1232];
            double R_1_1_1_1 = xpq * R_2_0_1_1;
            double R_0_2_1_1 = xpq * R_1_1_1_1 + 1 * R_1_0_1_1;
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+288];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[tx+608];
            vj_kl[2] -= R_0_2_1_1 * dm_ij_cache[tx+928];
            vj_kl[3] -= R_0_2_1_1 * dm_ij_cache[tx+1248];
            double R_2_1_0_1 = xpq * R_3_0_0_1;
            double R_1_2_0_1 = xpq * R_2_1_0_1 + 1 * R_2_0_0_1;
            double R_0_3_0_1 = xpq * R_1_2_0_1 + 2 * R_1_1_0_1;
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[tx+304];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[tx+624];
            vj_kl[2] -= R_0_3_0_1 * dm_ij_cache[tx+944];
            vj_kl[3] -= R_0_3_0_1 * dm_ij_cache[tx+1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+176] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_0_1_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_0_1_0 * dm_ij_cache[tx+320];
            vj_kl[2] -= R_0_0_1_0 * dm_ij_cache[tx+640];
            vj_kl[3] -= R_0_0_1_0 * dm_ij_cache[tx+960];
            vj_kl[0] -= R_0_0_1_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_0_1_1 * dm_ij_cache[tx+336];
            vj_kl[2] -= R_0_0_1_1 * dm_ij_cache[tx+656];
            vj_kl[3] -= R_0_0_1_1 * dm_ij_cache[tx+976];
            vj_kl[0] -= R_0_0_1_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_0_1_2 * dm_ij_cache[tx+352];
            vj_kl[2] -= R_0_0_1_2 * dm_ij_cache[tx+672];
            vj_kl[3] -= R_0_0_1_2 * dm_ij_cache[tx+992];
            vj_kl[0] -= R_0_0_1_3 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_0_1_3 * dm_ij_cache[tx+368];
            vj_kl[2] -= R_0_0_1_3 * dm_ij_cache[tx+688];
            vj_kl[3] -= R_0_0_1_3 * dm_ij_cache[tx+1008];
            vj_kl[0] -= R_0_0_2_0 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_0_2_0 * dm_ij_cache[tx+384];
            vj_kl[2] -= R_0_0_2_0 * dm_ij_cache[tx+704];
            vj_kl[3] -= R_0_0_2_0 * dm_ij_cache[tx+1024];
            vj_kl[0] -= R_0_0_2_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_0_2_1 * dm_ij_cache[tx+400];
            vj_kl[2] -= R_0_0_2_1 * dm_ij_cache[tx+720];
            vj_kl[3] -= R_0_0_2_1 * dm_ij_cache[tx+1040];
            vj_kl[0] -= R_0_0_2_2 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_0_2_2 * dm_ij_cache[tx+416];
            vj_kl[2] -= R_0_0_2_2 * dm_ij_cache[tx+736];
            vj_kl[3] -= R_0_0_2_2 * dm_ij_cache[tx+1056];
            vj_kl[0] -= R_0_0_3_0 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_0_3_0 * dm_ij_cache[tx+432];
            vj_kl[2] -= R_0_0_3_0 * dm_ij_cache[tx+752];
            vj_kl[3] -= R_0_0_3_0 * dm_ij_cache[tx+1072];
            vj_kl[0] -= R_0_0_3_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_0_3_1 * dm_ij_cache[tx+448];
            vj_kl[2] -= R_0_0_3_1 * dm_ij_cache[tx+768];
            vj_kl[3] -= R_0_0_3_1 * dm_ij_cache[tx+1088];
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_0_3_0 = ypq * R_2_0_2_0 + 2 * R_2_0_1_0;
            double R_0_0_4_0 = ypq * R_1_0_3_0 + 3 * R_1_0_2_0;
            vj_kl[0] -= R_0_0_4_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_0_4_0 * dm_ij_cache[tx+464];
            vj_kl[2] -= R_0_0_4_0 * dm_ij_cache[tx+784];
            vj_kl[3] -= R_0_0_4_0 * dm_ij_cache[tx+1104];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+160];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+480];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+800];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+1120];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+176];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+496];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+816];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+1136];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+192];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[tx+512];
            vj_kl[2] -= R_0_1_1_2 * dm_ij_cache[tx+832];
            vj_kl[3] -= R_0_1_1_2 * dm_ij_cache[tx+1152];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+208];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+528];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[tx+848];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[tx+1168];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+224];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[tx+544];
            vj_kl[2] -= R_0_1_2_1 * dm_ij_cache[tx+864];
            vj_kl[3] -= R_0_1_2_1 * dm_ij_cache[tx+1184];
            double R_0_1_3_0 = xpq * R_1_0_3_0;
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[tx+240];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[tx+560];
            vj_kl[2] -= R_0_1_3_0 * dm_ij_cache[tx+880];
            vj_kl[3] -= R_0_1_3_0 * dm_ij_cache[tx+1200];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+256];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+576];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[tx+896];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[tx+1216];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+272];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[tx+592];
            vj_kl[2] -= R_0_2_1_1 * dm_ij_cache[tx+912];
            vj_kl[3] -= R_0_2_1_1 * dm_ij_cache[tx+1232];
            double R_1_1_2_0 = xpq * R_2_0_2_0;
            double R_0_2_2_0 = xpq * R_1_1_2_0 + 1 * R_1_0_2_0;
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[tx+288];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[tx+608];
            vj_kl[2] -= R_0_2_2_0 * dm_ij_cache[tx+928];
            vj_kl[3] -= R_0_2_2_0 * dm_ij_cache[tx+1248];
            double R_2_1_1_0 = xpq * R_3_0_1_0;
            double R_1_2_1_0 = xpq * R_2_1_1_0 + 1 * R_2_0_1_0;
            double R_0_3_1_0 = xpq * R_1_2_1_0 + 2 * R_1_1_1_0;
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[tx+304];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[tx+624];
            vj_kl[2] -= R_0_3_1_0 * dm_ij_cache[tx+944];
            vj_kl[3] -= R_0_3_1_0 * dm_ij_cache[tx+1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+352] += vj_kl[m];
            } }
            for (int m = 0; m < 4; ++m) vj_kl[m] = 0.;
            vj_kl[0] -= R_0_1_0_0 * dm_ij_cache[tx+0];
            vj_kl[1] -= R_0_1_0_0 * dm_ij_cache[tx+320];
            vj_kl[2] -= R_0_1_0_0 * dm_ij_cache[tx+640];
            vj_kl[3] -= R_0_1_0_0 * dm_ij_cache[tx+960];
            vj_kl[0] -= R_0_1_0_1 * dm_ij_cache[tx+16];
            vj_kl[1] -= R_0_1_0_1 * dm_ij_cache[tx+336];
            vj_kl[2] -= R_0_1_0_1 * dm_ij_cache[tx+656];
            vj_kl[3] -= R_0_1_0_1 * dm_ij_cache[tx+976];
            vj_kl[0] -= R_0_1_0_2 * dm_ij_cache[tx+32];
            vj_kl[1] -= R_0_1_0_2 * dm_ij_cache[tx+352];
            vj_kl[2] -= R_0_1_0_2 * dm_ij_cache[tx+672];
            vj_kl[3] -= R_0_1_0_2 * dm_ij_cache[tx+992];
            vj_kl[0] -= R_0_1_0_3 * dm_ij_cache[tx+48];
            vj_kl[1] -= R_0_1_0_3 * dm_ij_cache[tx+368];
            vj_kl[2] -= R_0_1_0_3 * dm_ij_cache[tx+688];
            vj_kl[3] -= R_0_1_0_3 * dm_ij_cache[tx+1008];
            vj_kl[0] -= R_0_1_1_0 * dm_ij_cache[tx+64];
            vj_kl[1] -= R_0_1_1_0 * dm_ij_cache[tx+384];
            vj_kl[2] -= R_0_1_1_0 * dm_ij_cache[tx+704];
            vj_kl[3] -= R_0_1_1_0 * dm_ij_cache[tx+1024];
            vj_kl[0] -= R_0_1_1_1 * dm_ij_cache[tx+80];
            vj_kl[1] -= R_0_1_1_1 * dm_ij_cache[tx+400];
            vj_kl[2] -= R_0_1_1_1 * dm_ij_cache[tx+720];
            vj_kl[3] -= R_0_1_1_1 * dm_ij_cache[tx+1040];
            vj_kl[0] -= R_0_1_1_2 * dm_ij_cache[tx+96];
            vj_kl[1] -= R_0_1_1_2 * dm_ij_cache[tx+416];
            vj_kl[2] -= R_0_1_1_2 * dm_ij_cache[tx+736];
            vj_kl[3] -= R_0_1_1_2 * dm_ij_cache[tx+1056];
            vj_kl[0] -= R_0_1_2_0 * dm_ij_cache[tx+112];
            vj_kl[1] -= R_0_1_2_0 * dm_ij_cache[tx+432];
            vj_kl[2] -= R_0_1_2_0 * dm_ij_cache[tx+752];
            vj_kl[3] -= R_0_1_2_0 * dm_ij_cache[tx+1072];
            vj_kl[0] -= R_0_1_2_1 * dm_ij_cache[tx+128];
            vj_kl[1] -= R_0_1_2_1 * dm_ij_cache[tx+448];
            vj_kl[2] -= R_0_1_2_1 * dm_ij_cache[tx+768];
            vj_kl[3] -= R_0_1_2_1 * dm_ij_cache[tx+1088];
            vj_kl[0] -= R_0_1_3_0 * dm_ij_cache[tx+144];
            vj_kl[1] -= R_0_1_3_0 * dm_ij_cache[tx+464];
            vj_kl[2] -= R_0_1_3_0 * dm_ij_cache[tx+784];
            vj_kl[3] -= R_0_1_3_0 * dm_ij_cache[tx+1104];
            vj_kl[0] -= R_0_2_0_0 * dm_ij_cache[tx+160];
            vj_kl[1] -= R_0_2_0_0 * dm_ij_cache[tx+480];
            vj_kl[2] -= R_0_2_0_0 * dm_ij_cache[tx+800];
            vj_kl[3] -= R_0_2_0_0 * dm_ij_cache[tx+1120];
            vj_kl[0] -= R_0_2_0_1 * dm_ij_cache[tx+176];
            vj_kl[1] -= R_0_2_0_1 * dm_ij_cache[tx+496];
            vj_kl[2] -= R_0_2_0_1 * dm_ij_cache[tx+816];
            vj_kl[3] -= R_0_2_0_1 * dm_ij_cache[tx+1136];
            vj_kl[0] -= R_0_2_0_2 * dm_ij_cache[tx+192];
            vj_kl[1] -= R_0_2_0_2 * dm_ij_cache[tx+512];
            vj_kl[2] -= R_0_2_0_2 * dm_ij_cache[tx+832];
            vj_kl[3] -= R_0_2_0_2 * dm_ij_cache[tx+1152];
            vj_kl[0] -= R_0_2_1_0 * dm_ij_cache[tx+208];
            vj_kl[1] -= R_0_2_1_0 * dm_ij_cache[tx+528];
            vj_kl[2] -= R_0_2_1_0 * dm_ij_cache[tx+848];
            vj_kl[3] -= R_0_2_1_0 * dm_ij_cache[tx+1168];
            vj_kl[0] -= R_0_2_1_1 * dm_ij_cache[tx+224];
            vj_kl[1] -= R_0_2_1_1 * dm_ij_cache[tx+544];
            vj_kl[2] -= R_0_2_1_1 * dm_ij_cache[tx+864];
            vj_kl[3] -= R_0_2_1_1 * dm_ij_cache[tx+1184];
            vj_kl[0] -= R_0_2_2_0 * dm_ij_cache[tx+240];
            vj_kl[1] -= R_0_2_2_0 * dm_ij_cache[tx+560];
            vj_kl[2] -= R_0_2_2_0 * dm_ij_cache[tx+880];
            vj_kl[3] -= R_0_2_2_0 * dm_ij_cache[tx+1200];
            vj_kl[0] -= R_0_3_0_0 * dm_ij_cache[tx+256];
            vj_kl[1] -= R_0_3_0_0 * dm_ij_cache[tx+576];
            vj_kl[2] -= R_0_3_0_0 * dm_ij_cache[tx+896];
            vj_kl[3] -= R_0_3_0_0 * dm_ij_cache[tx+1216];
            vj_kl[0] -= R_0_3_0_1 * dm_ij_cache[tx+272];
            vj_kl[1] -= R_0_3_0_1 * dm_ij_cache[tx+592];
            vj_kl[2] -= R_0_3_0_1 * dm_ij_cache[tx+912];
            vj_kl[3] -= R_0_3_0_1 * dm_ij_cache[tx+1232];
            vj_kl[0] -= R_0_3_1_0 * dm_ij_cache[tx+288];
            vj_kl[1] -= R_0_3_1_0 * dm_ij_cache[tx+608];
            vj_kl[2] -= R_0_3_1_0 * dm_ij_cache[tx+928];
            vj_kl[3] -= R_0_3_1_0 * dm_ij_cache[tx+1248];
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
            double R_1_3_0_0 = xpq * R_2_2_0_0 + 2 * R_2_1_0_0;
            double R_0_4_0_0 = xpq * R_1_3_0_0 + 3 * R_1_2_0_0;
            vj_kl[0] -= R_0_4_0_0 * dm_ij_cache[tx+304];
            vj_kl[1] -= R_0_4_0_0 * dm_ij_cache[tx+624];
            vj_kl[2] -= R_0_4_0_0 * dm_ij_cache[tx+944];
            vj_kl[3] -= R_0_4_0_0 * dm_ij_cache[tx+1264];
            for (int offset = 8; offset > 0; offset /= 2) {
            for (int m = 0; m < 4; ++m) {
                vj_kl[m] += __shfl_down_sync(mask, vj_kl[m], offset);
            } }
            if (tx == 0) {
            for (int m = 0; m < 4; ++m) {
                vj_kl_cache[sq_kl+m*704+528] += vj_kl[m];
            } }
            }{
            for (int m = 0; m < 4; ++m) { dm_kl[m] = 1 * dm[m*dm_size+kl_loc0+0]; }
            vj_ij[0] += gamma_inc[sq_id+0*256] * dm_kl[0];
            vj_ij[20] += gamma_inc[sq_id+0*256] * dm_kl[1];
            vj_ij[40] += gamma_inc[sq_id+0*256] * dm_kl[2];
            vj_ij[60] += gamma_inc[sq_id+0*256] * dm_kl[3];
            double R_0_0_0_1 = zpq * gamma_inc[sq_id+1*256];
            vj_ij[1] += R_0_0_0_1 * dm_kl[0];
            vj_ij[21] += R_0_0_0_1 * dm_kl[1];
            vj_ij[41] += R_0_0_0_1 * dm_kl[2];
            vj_ij[61] += R_0_0_0_1 * dm_kl[3];
            double R_1_0_0_1 = zpq * gamma_inc[sq_id+2*256];
            double R_0_0_0_2 = zpq * R_1_0_0_1 + 1 * gamma_inc[sq_id+1*256];
            vj_ij[2] += R_0_0_0_2 * dm_kl[0];
            vj_ij[22] += R_0_0_0_2 * dm_kl[1];
            vj_ij[42] += R_0_0_0_2 * dm_kl[2];
            vj_ij[62] += R_0_0_0_2 * dm_kl[3];
            double R_2_0_0_1 = zpq * gamma_inc[sq_id+3*256];
            double R_1_0_0_2 = zpq * R_2_0_0_1 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_0_3 = zpq * R_1_0_0_2 + 2 * R_1_0_0_1;
            vj_ij[3] += R_0_0_0_3 * dm_kl[0];
            vj_ij[23] += R_0_0_0_3 * dm_kl[1];
            vj_ij[43] += R_0_0_0_3 * dm_kl[2];
            vj_ij[63] += R_0_0_0_3 * dm_kl[3];
            double R_0_0_1_0 = ypq * gamma_inc[sq_id+1*256];
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
            double R_1_0_1_0 = ypq * gamma_inc[sq_id+2*256];
            double R_0_0_2_0 = ypq * R_1_0_1_0 + 1 * gamma_inc[sq_id+1*256];
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
            double R_2_0_1_0 = ypq * gamma_inc[sq_id+3*256];
            double R_1_0_2_0 = ypq * R_2_0_1_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_0_3_0 = ypq * R_1_0_2_0 + 2 * R_1_0_1_0;
            vj_ij[9] += R_0_0_3_0 * dm_kl[0];
            vj_ij[29] += R_0_0_3_0 * dm_kl[1];
            vj_ij[49] += R_0_0_3_0 * dm_kl[2];
            vj_ij[69] += R_0_0_3_0 * dm_kl[3];
            double R_0_1_0_0 = xpq * gamma_inc[sq_id+1*256];
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
            double R_1_1_0_0 = xpq * gamma_inc[sq_id+2*256];
            double R_0_2_0_0 = xpq * R_1_1_0_0 + 1 * gamma_inc[sq_id+1*256];
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
            double R_2_1_0_0 = xpq * gamma_inc[sq_id+3*256];
            double R_1_2_0_0 = xpq * R_2_1_0_0 + 1 * gamma_inc[sq_id+2*256];
            double R_0_3_0_0 = xpq * R_1_2_0_0 + 2 * R_1_1_0_0;
            vj_ij[19] += R_0_3_0_0 * dm_kl[0];
            vj_ij[39] += R_0_3_0_0 * dm_kl[1];
            vj_ij[59] += R_0_3_0_0 * dm_kl[2];
            vj_ij[79] += R_0_3_0_0 * dm_kl[3];
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+1]; }
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
            double R_3_0_0_1 = zpq * gamma_inc[sq_id+4*256];
            double R_2_0_0_2 = zpq * R_3_0_0_1 + 1 * gamma_inc[sq_id+3*256];
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+2]; }
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
            double R_3_0_1_0 = ypq * gamma_inc[sq_id+4*256];
            double R_2_0_2_0 = ypq * R_3_0_1_0 + 1 * gamma_inc[sq_id+3*256];
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
            for (int m = 0; m < 4; ++m) { dm_kl[m] = -1 * dm[m*dm_size+kl_loc0+3]; }
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
            double R_3_1_0_0 = xpq * gamma_inc[sq_id+4*256];
            double R_2_2_0_0 = xpq * R_3_1_0_0 + 1 * gamma_inc[sq_id+3*256];
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
        } else if (remaining_n_dm == 2) {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    atomicAdd(vj        +ij_loc0+n, vj_cache [thread_id]);
                    atomicAdd(vj+dm_size+ij_loc0+n, vj_cache1[thread_id]);
                }
            }
        } else {
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
                if (ty == 0 && task_ij0+tx < npairs_ij) {
                    for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                        atomicAdd(vj+m*dm_size +ij_loc0+n, vj_cache[thread_id+256*m]);
                    }
                }
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
            for (int m = 0; m < min(4, remaining_n_dm); ++m) {
                atomicAdd(vj+m*dm_size+kl_loc0+kl, vj_kl_cache[sq_kl+m*704+kl*176]);
            }
        }
    }
} }

int md_j_4dm_unrolled(RysIntEnvVars *envs, JKMatrix *jk, MDBoundsInfo *bounds, int dm_size)
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
    case 0: { // lij=0, lkl=0, tilex=21, tiley=21
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 335) / 336, (npairs_kl + 335) / 336, 1);
        md_j_4dm_0_0<<<blocks, threads, 6080*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 9: { // lij=1, lkl=0, tilex=48, tiley=21
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 335) / 336, 1);
        md_j_4dm_1_0<<<blocks, threads, 6080*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 10: { // lij=1, lkl=1, tilex=7, tiley=7
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 111) / 112, (npairs_kl + 111) / 112, 1);
        md_j_4dm_1_1<<<blocks, threads, 6080*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 18: { // lij=2, lkl=0, tilex=48, tiley=21
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 335) / 336, 1);
        md_j_4dm_2_0<<<blocks, threads, 6144*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 19: { // lij=2, lkl=1, tilex=48, tiley=13
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 207) / 208, 1);
        md_j_4dm_2_1<<<blocks, threads, 5888*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 20: { // lij=2, lkl=2, tilex=5, tiley=5
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 79) / 80, (npairs_kl + 79) / 80, 1);
        md_j_4dm_2_2<<<blocks, threads, 5504*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 27: { // lij=3, lkl=0, tilex=48, tiley=29
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 463) / 464, 1);
        md_j_4dm_3_0<<<blocks, threads, 6080*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    case 28: { // lij=3, lkl=1, tilex=48, tiley=11
        dim3 threads(16, 16);
        dim3 blocks((npairs_ij + 767) / 768, (npairs_kl + 175) / 176, 1);
        md_j_4dm_3_1<<<blocks, threads, 6144*sizeof(double)>>>(*envs, *jk, *bounds, dm_size);
    } break;
    default: return 0;
    }
    return 1;
}
