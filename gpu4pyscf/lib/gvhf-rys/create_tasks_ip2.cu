#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "vhf.cuh"

__device__
static int _fill_ejk_ip2_type2_tasks(ShellQuartet *shl_quartet_idx,
                           RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                           int batch_ij, int batch_kl)
{
    int nbas = envs.nbas;
    int *tile_ij_mapping = bounds.tile_ij_mapping;
    int *tile_kl_mapping = bounds.tile_kl_mapping;
    float *q_cond = bounds.q_cond;
    float *tile_q_cond = bounds.tile_q_cond;
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
                float q_ij = q_cond [ish*nbas+jsh];
                float d_ij = dm_cond[ish*nbas+jsh];
                for (int ksh = ksh0; ksh < ksh1; ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < MIN(ksh+1, lsh1); ++lsh) {
                        float q_ijkl = q_ij + q_cond[ksh*nbas+lsh];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik+dm_cond[jsh*nbas+lsh] > d_cutoff ||
                                      d_jk+dm_cond[ish*nbas+lsh] > d_cutoff)) ||
                            (do_j && d_ij+dm_cond[ksh*nbas+lsh] > d_cutoff)) {
                            ++count;
                        }
                    }
                }
            }
        }
    }

    extern __shared__ int cum_count[];
    cum_count[t_id] = count;
    // Up-sweep phase
    for (int stride = 1; stride < threads; stride *= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index < threads) {
            cum_count[index] += cum_count[index-stride];
        }
    }
    __syncthreads();
    // Down-sweep phase
    for (int stride = threads/4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index + stride < threads) {
            cum_count[index + stride] += cum_count[index];
        }
    }
    __syncthreads();
    int ntasks = cum_count[threads-1];
    if (ntasks == 0) {
        return ntasks;
    }

    int offset = 0;
    if (t_id > 0) {
        offset = cum_count[t_id-1];
    }
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
                float q_ij = q_cond [ish*nbas+jsh];
                float d_ij = dm_cond[ish*nbas+jsh];
                sq.i = ish;
                sq.j = jsh;
                for (int ksh = ksh0; ksh < ksh1; ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < MIN(ksh+1, lsh1); ++lsh) {
                        float q_ijkl = q_ij + q_cond[ksh*nbas+lsh];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik+dm_cond[jsh*nbas+lsh] > d_cutoff ||
                                      d_jk+dm_cond[ish*nbas+lsh] > d_cutoff)) ||
                            (do_j && d_ij+dm_cond[ksh*nbas+lsh] > d_cutoff)) {
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
static int _fill_ejk_ip2_type3_tasks(ShellQuartet *shl_quartet_idx,
                           RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                           int batch_ij, int batch_kl)
{
    int nbas = envs.nbas;
    int *tile_ij_mapping = bounds.tile_ij_mapping;
    int *tile_kl_mapping = bounds.tile_kl_mapping;
    float *q_cond = bounds.q_cond;
    float *tile_q_cond = bounds.tile_q_cond;
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
            for (int jsh = jsh0; jsh < jsh1; ++jsh) {
                float q_ij = q_cond [ish*nbas+jsh];
                float d_ij = dm_cond[ish*nbas+jsh];
                int bas_ij = ish * nbas + jsh;
                for (int ksh = ksh0; ksh < MIN(ish+1, ksh1); ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < lsh1; ++lsh) {
                        int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) {
                            continue;
                        }
                        float q_ijkl = q_ij + q_cond[ksh*nbas+lsh];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik+dm_cond[jsh*nbas+lsh] > d_cutoff ||
                                      d_jk+dm_cond[ish*nbas+lsh] > d_cutoff)) ||
                            (do_j && d_ij+dm_cond[ksh*nbas+lsh] > d_cutoff)) {
                            ++count;
                        }
                    }
                }
            }
        }
    }

    extern __shared__ int cum_count[];
    cum_count[t_id] = count;
    // Up-sweep phase
    for (int stride = 1; stride < threads; stride *= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index < threads) {
            cum_count[index] += cum_count[index-stride];
        }
    }
    __syncthreads();
    // Down-sweep phase
    for (int stride = threads/4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (t_id + 1) * stride * 2 - 1;
        if (index + stride < threads) {
            cum_count[index + stride] += cum_count[index];
        }
    }
    __syncthreads();
    int ntasks = cum_count[threads-1];
    if (ntasks == 0) {
        return ntasks;
    }

    int offset = 0;
    if (t_id > 0) {
        offset = cum_count[t_id-1];
    }
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
            for (int jsh = jsh0; jsh < jsh1; ++jsh) {
                float q_ij = q_cond [ish*nbas+jsh];
                float d_ij = dm_cond[ish*nbas+jsh];
                int bas_ij = ish * nbas + jsh;
                sq.i = ish;
                sq.j = jsh;
                for (int ksh = ksh0; ksh < MIN(ish+1, ksh1); ++ksh) {
                    float d_ik = dm_cond[ish*nbas+ksh];
                    float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int lsh = lsh0; lsh < lsh1; ++lsh) {
                        int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) {
                            continue;
                        }
                        float q_ijkl = q_ij + q_cond[ksh*nbas+lsh];
                        if (q_ijkl < cutoff) {
                            continue;
                        }
                        float d_cutoff = cutoff - q_ijkl;
                        if ((do_k && (d_ik+dm_cond[jsh*nbas+lsh] > d_cutoff ||
                                      d_jk+dm_cond[ish*nbas+lsh] > d_cutoff)) ||
                            (do_j && d_ij+dm_cond[ksh*nbas+lsh] > d_cutoff)) {
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
