#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "vhf.cuh"

__device__
static int _fill_jk_tasks(ShellQuartet *shl_quartet_idx,
                          RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                          int batch_ij, int batch_kl)
{
    int nbas = envs.nbas;
    int *tile_ij_mapping = bounds.tile_ij_mapping;
    int *tile_kl_mapping = bounds.tile_kl_mapping;
    float *q_cond = bounds.q_cond;
#if TILE == 1
    float *tile_q_cond = q_cond;
#else
    float *tile_q_cond = q_cond + nbas*nbas;
#endif
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int t_kl0 = batch_kl * TILES_IN_BATCH;
    int t_kl1 = MIN(t_kl0 + TILES_IN_BATCH, bounds.ntile_kl_pairs);
    int threads = blockDim.x * blockDim.y;

    int tile_ij = tile_ij_mapping[batch_ij];
    int nbas_tiles = nbas / TILE;
    int tile_i = tile_ij / nbas_tiles;
    int tile_j = tile_ij % nbas_tiles;
    int ish0 = tile_i * TILE;
    int jsh0 = tile_j * TILE;
    int ish1 = ish0 + TILE;
    int jsh1 = jsh0 + TILE;
    int do_j = jk.vj != NULL;
    int do_k = jk.vk != NULL;

    int count = 0;
    float tile_q_ij = tile_q_cond[tile_ij];
    for (int t_kl_id = t_kl0+t_id; t_kl_id < t_kl1; t_kl_id += threads) {
        int tile_kl = tile_kl_mapping[t_kl_id];
        if (tile_q_ij + tile_q_cond[tile_kl] < cutoff) {
            break;
        }
        int tile_k = tile_kl / nbas_tiles;
        int tile_l = tile_kl % nbas_tiles;
        int ksh0 = tile_k * TILE;
        int lsh0 = tile_l * TILE;
        int ksh1 = ksh0 + TILE;
        int lsh1 = lsh0 + TILE;
        for (int ish = ish0; ish < ish1; ++ish) {
            for (int jsh = jsh0; jsh < MIN(ish+1, jsh1); ++jsh) {
                int bas_ij = ish * nbas + jsh;
                float q_ij = q_cond [bas_ij];
                float d_ij = dm_cond[bas_ij];
                for (int ksh = ksh0; ksh < MIN(ish+1, ksh1); ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < MIN(ksh+1, lsh1); ++lsh) {
                        int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) {
                            continue;
                        }
                        float q_ijkl = q_ij + q_cond[bas_kl];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik                  > d_cutoff ||
                                      d_jk                  > d_cutoff ||
                                      dm_cond[ish*nbas+lsh] > d_cutoff ||
                                      dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
                            (do_j && (d_ij                  > d_cutoff ||
                                      dm_cond[bas_kl      ] > d_cutoff))) {
                            ++count;
                        }
                    }
                }
            }
        }
    }

    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    extern __shared__ int thread_offsets[];
    thread_offsets[t_id] = count;
    // Up-sweep phase
    for (int stride = 1; stride < threads; stride *= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index < threads) {
            thread_offsets[index] += thread_offsets[index-stride];
        }
    }
    __syncthreads();
    if (t_id == threads-1) { thread_offsets[threads-1] = 0; }
    // Down-sweep phase
    for (int stride = threads/2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index < threads) {
            int temp = thread_offsets[index - stride];
            thread_offsets[index - stride] = thread_offsets[index];
            thread_offsets[index] += temp;
        }
    }
    __syncthreads();
    __shared__ int ntasks;
    if (t_id == threads-1) {
        ntasks = thread_offsets[threads-1] + count;
    }
    __syncthreads();
    if (ntasks == 0) {
        return ntasks;
    }

    int offset = thread_offsets[t_id];
    for (int t_kl_id = t_kl0+t_id; t_kl_id < t_kl1; t_kl_id += threads) {
        int tile_kl = tile_kl_mapping[t_kl_id];
        if (tile_q_ij + tile_q_cond[tile_kl] < cutoff) {
            break;
        }
        int tile_k = tile_kl / nbas_tiles;
        int tile_l = tile_kl % nbas_tiles;
        int ksh0 = tile_k * TILE;
        int lsh0 = tile_l * TILE;
        int ksh1 = ksh0 + TILE;
        int lsh1 = lsh0 + TILE;
        ShellQuartet sq;
        for (int ish = ish0; ish < ish1; ++ish) {
            for (int jsh = jsh0; jsh < MIN(ish+1, jsh1); ++jsh) {
                int bas_ij = ish * nbas + jsh;
                float q_ij = q_cond [bas_ij];
                float d_ij = dm_cond[bas_ij];
                sq.i = ish;
                sq.j = jsh;
                for (int ksh = ksh0; ksh < MIN(ish+1, ksh1); ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < MIN(ksh+1, lsh1); ++lsh) {
                        int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) {
                            continue;
                        }
                        float q_ijkl = q_ij + q_cond[bas_kl];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik                  > d_cutoff ||
                                      d_jk                  > d_cutoff ||
                                      dm_cond[ish*nbas+lsh] > d_cutoff ||
                                      dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
                            (do_j && (d_ij                  > d_cutoff ||
                                      dm_cond[bas_kl      ] > d_cutoff))) {
                            sq.k = ksh;
                            sq.l = lsh;
                            shl_quartet_idx[offset] = sq;
                            ++offset;
                        }
                    }
                }
            }
        }
    }
    return ntasks;
}

__device__
static int _fill_sr_jk_tasks(ShellQuartet *shl_quartet_idx,
                             RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                             int batch_ij, int batch_kl)
{
    int nbas = envs.nbas;
    int *tile_ij_mapping = bounds.tile_ij_mapping;
    int *tile_kl_mapping = bounds.tile_kl_mapping;
    float *q_cond = bounds.q_cond;
#if TILE == 1
    float *tile_q_cond = q_cond;
#else
    float *tile_q_cond = q_cond + nbas*nbas;
#endif
    int nbas_tiles = nbas / TILE;
    // TODO: implement q_ijij_cond
    float *s_estimator = tile_q_cond + nbas_tiles*nbas_tiles;
    float *dm_cond = bounds.dm_cond;
    float cutoff = bounds.cutoff;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    int t_kl0 = batch_kl * TILES_IN_BATCH;
    int t_kl1 = MIN(t_kl0 + TILES_IN_BATCH, bounds.ntile_kl_pairs);
    int threads = blockDim.x * blockDim.y;

    int tile_ij = tile_ij_mapping[batch_ij];
    int tile_i = tile_ij / nbas_tiles;
    int tile_j = tile_ij % nbas_tiles;
    int ish0 = tile_i * TILE;
    int jsh0 = tile_j * TILE;
    int ish1 = ish0 + TILE;
    int jsh1 = jsh0 + TILE;
    int do_j = jk.vj != NULL;
    int do_k = jk.vk != NULL;

    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *bas = envs.bas;
    double *env = envs.env;
    float omega = env[PTR_RANGE_OMEGA];
    float omega2 = omega * omega;

    int count = 0;
    float tile_q_ij = tile_q_cond[tile_ij];
    for (int t_kl_id = t_kl0+t_id; t_kl_id < t_kl1; t_kl_id += threads) {
        int tile_kl = tile_kl_mapping[t_kl_id];
        if (tile_q_ij + tile_q_cond[tile_kl] < cutoff) {
            break;
        }
        int tile_k = tile_kl / nbas_tiles;
        int tile_l = tile_kl % nbas_tiles;
        int ksh0 = tile_k * TILE;
        int lsh0 = tile_l * TILE;
        int ksh1 = ksh0 + TILE;
        int lsh1 = lsh0 + TILE;
        for (int ish = ish0; ish < ish1; ++ish) {
            for (int jsh = jsh0; jsh < MIN(ish+1, jsh1); ++jsh) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                float ai = expi[iprim-1];
                float aj = expj[jprim-1];
                float aij = ai + aj;
                float aj_aij = aj / aij;
                float xi = ri[0];
                float yi = ri[1];
                float zi = ri[2];
                float xj = rj[0];
                float yj = rj[1];
                float zj = rj[2];
                float xjxi = xj - xi;
                float yjyi = yj - yi;
                float zjzi = zj - zi;
                float xpa = xjxi * aj_aij;
                float ypa = yjyi * aj_aij;
                float zpa = zjzi * aj_aij;
                float xij = xi + xpa;
                float yij = yi + ypa;
                float zij = zi + zpa;
                int bas_ij = ish * nbas + jsh;
                float q_ij = q_cond [bas_ij];
                float d_ij = dm_cond[bas_ij];
                float skl_cutoff = cutoff - s_estimator[bas_ij];
                for (int ksh = ksh0; ksh < MIN(ish+1, ksh1); ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < MIN(ksh+1, lsh1); ++lsh) {
                        int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) {
                            continue;
                        }
                        float q_ijkl = q_ij + q_cond[bas_kl];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik                  > d_cutoff ||
                                      d_jk                  > d_cutoff ||
                                      dm_cond[ish*nbas+lsh] > d_cutoff ||
                                      dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
                            (do_j && (d_ij                  > d_cutoff ||
                                      dm_cond[bas_kl      ] > d_cutoff))) {

                            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
                            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                            float ak = expk[kprim-1];
                            float al = expl[lprim-1];
                            float akl = ak + al;
                            float al_akl = al / akl;
                            float xk = rk[0];
                            float yk = rk[1];
                            float zk = rk[2];
                            float xl = rl[0];
                            float yl = rl[1];
                            float zl = rl[2];
                            float xlxk = xl - xk;
                            float ylyk = yl - yk;
                            float zlzk = zl - zk;
                            float xqc = xlxk * al_akl;
                            float yqc = ylyk * al_akl;
                            float zqc = zlzk * al_akl;
                            float xkl = xk + xqc;
                            float ykl = yk + yqc;
                            float zkl = zk + zqc;
                            float theta = 1./(1./aij+1./akl+1./omega2);
                            float xpq = xij - xkl;
                            float ypq = yij - ykl;
                            float zpq = zij - zkl;
                            float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                            float theta_rr = logf(rr + 1e-30f) + theta * rr;
                            d_cutoff = skl_cutoff - s_estimator[bas_kl] + theta_rr;
                            if (d_cutoff > 0) {
                                continue;
                            }
                            if ((do_k && (d_ik                  > d_cutoff ||
                                          d_jk                  > d_cutoff ||
                                          dm_cond[ish*nbas+lsh] > d_cutoff ||
                                          dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
                                (do_j && (d_ij                  > d_cutoff ||
                                          dm_cond[bas_kl      ] > d_cutoff))) {
                                ++count;
                            }
                        }
                    }
                }
            }
        }
    }

    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
    extern __shared__ int thread_offsets[];
    thread_offsets[t_id] = count;
    // Up-sweep phase
    for (int stride = 1; stride < threads; stride *= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index < threads) {
            thread_offsets[index] += thread_offsets[index-stride];
        }
    }
    __syncthreads();
    if (t_id == threads-1) { thread_offsets[threads-1] = 0; }
    // Down-sweep phase
    for (int stride = threads/2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index < threads) {
            int temp = thread_offsets[index - stride];
            thread_offsets[index - stride] = thread_offsets[index];
            thread_offsets[index] += temp;
        }
    }
    __syncthreads();
    __shared__ int ntasks;
    if (t_id == threads-1) {
        ntasks = thread_offsets[threads-1] + count;
    }
    __syncthreads();
    if (ntasks == 0) {
        return ntasks;
    }

    int offset = thread_offsets[t_id];
    for (int t_kl_id = t_kl0+t_id; t_kl_id < t_kl1; t_kl_id += threads) {
        int tile_kl = tile_kl_mapping[t_kl_id];
        if (tile_q_ij + tile_q_cond[tile_kl] < cutoff) {
            break;
        }
        int tile_k = tile_kl / nbas_tiles;
        int tile_l = tile_kl % nbas_tiles;
        int ksh0 = tile_k * TILE;
        int lsh0 = tile_l * TILE;
        int ksh1 = ksh0 + TILE;
        int lsh1 = lsh0 + TILE;
        ShellQuartet sq;
        for (int ish = ish0; ish < ish1; ++ish) {
            for (int jsh = jsh0; jsh < MIN(ish+1, jsh1); ++jsh) {
                sq.i = ish;
                sq.j = jsh;
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                float ai = expi[iprim-1];
                float aj = expj[jprim-1];
                float aij = ai + aj;
                float aj_aij = aj / aij;
                float xi = ri[0];
                float yi = ri[1];
                float zi = ri[2];
                float xj = rj[0];
                float yj = rj[1];
                float zj = rj[2];
                float xjxi = xj - xi;
                float yjyi = yj - yi;
                float zjzi = zj - zi;
                float xpa = xjxi * aj_aij;
                float ypa = yjyi * aj_aij;
                float zpa = zjzi * aj_aij;
                float xij = xi + xpa;
                float yij = yi + ypa;
                float zij = zi + zpa;
                int bas_ij = ish * nbas + jsh;
                float q_ij = q_cond [bas_ij];
                float d_ij = dm_cond[bas_ij];
                float skl_cutoff = cutoff - s_estimator[bas_ij];
                for (int ksh = ksh0; ksh < MIN(ish+1, ksh1); ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < MIN(ksh+1, lsh1); ++lsh) {
                        int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) {
                            continue;
                        }
                        float q_ijkl = q_ij + q_cond[bas_kl];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik                  > d_cutoff ||
                                      d_jk                  > d_cutoff ||
                                      dm_cond[ish*nbas+lsh] > d_cutoff ||
                                      dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
                            (do_j && (d_ij                  > d_cutoff ||
                                      dm_cond[bas_kl      ] > d_cutoff))) {

                            // TODO: cache in shared memory
                            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                            double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
                            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                            double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
                            float ak = expk[kprim-1];
                            float al = expl[lprim-1];
                            float akl = ak + al;
                            float al_akl = al / akl;
                            float xk = rk[0];
                            float yk = rk[1];
                            float zk = rk[2];
                            float xl = rl[0];
                            float yl = rl[1];
                            float zl = rl[2];
                            float xlxk = xl - xk;
                            float ylyk = yl - yk;
                            float zlzk = zl - zk;
                            float xqc = xlxk * al_akl;
                            float yqc = ylyk * al_akl;
                            float zqc = zlzk * al_akl;
                            float xkl = xk + xqc;
                            float ykl = yk + yqc;
                            float zkl = zk + zqc;
                            float theta = 1./(1./aij+1./akl+1./omega2);
                            float xpq = xij - xkl;
                            float ypq = yij - ykl;
                            float zpq = zij - zkl;
                            float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                            float theta_rr = logf(rr + 1e-30f) + theta * rr;
                            d_cutoff = skl_cutoff - s_estimator[bas_kl] + theta_rr;
                            if (d_cutoff > 0) {
                                continue;
                            }
                            if ((do_k && (d_ik                  > d_cutoff ||
                                          d_jk                  > d_cutoff ||
                                          dm_cond[ish*nbas+lsh] > d_cutoff ||
                                          dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
                                (do_j && (d_ij                  > d_cutoff ||
                                          dm_cond[bas_kl      ] > d_cutoff))) {
                                sq.k = ksh;
                                sq.l = lsh;
                                shl_quartet_idx[offset] = sq;
                                ++offset;
                            }
                        }
                    }
                }
            }
        }
    }
    return ntasks;
}
