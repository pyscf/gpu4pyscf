/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "vhf.cuh"

#define CHECK_SHARED_MEMORY_ATTRIBUTES true

__constant__ int c_g_pair_idx[3675];
__constant__ int c_g_pair_offsets[LMAX1*LMAX1];
// Putting _env in c_env reduces performance. Reason unclear
//__constant__ double c_env[6000];
// TODO: reuse memory of c_g_pair_idx for c_i_in_fold2idx and c_i_in_fold2idx
__constant__ Fold2Index c_i_in_fold2idx[165];
__constant__ Fold3Index c_i_in_fold3idx[495];

extern __global__ void rys_j_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                    ShellQuartet *pool, uint32_t *batch_head);
extern __global__ void rys_j_with_gout_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                    ShellQuartet *pool, uint32_t *batch_head);
extern __global__ void rys_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                     ShellQuartet *pool, uint32_t *batch_head);
extern __global__ void rys_jk_ip1_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                         ShellQuartet *pool, uint32_t *batch_head);
extern __global__ void rys_ejk_ip1_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                                          ShellQuartet *pool, double *dd_pool, uint32_t *batch_head);
extern __global__ void rys_ejk_ip2_type12_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                                          ShellQuartet *pool, double *dd_pool, uint32_t *batch_head);
extern __global__ void rys_ejk_ip2_type3_kernel(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                                          ShellQuartet *pool, double *dd_pool, uint32_t *batch_head);
extern int rys_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head, int *scheme, int workers);
extern int rys_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head, int *scheme, int workers);
extern int os_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head,
                    int *scheme, int workers, double omega);
extern int rys_vjk_ip1_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head, int *scheme, int workers);
extern int rys_ejk_ip1_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, double *dd_pool,
                    uint32_t *batch_head, int *scheme, int workers);
extern int rys_ejk_ip2_type12_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, double *dd_pool,
                    uint32_t *batch_head, int *scheme, int workers);
extern int rys_ejk_ip2_type3_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, double *dd_pool,
                    uint32_t *batch_head, int *scheme, int workers);

extern "C" {
int RYS_build_j(double *vj, double *dm, int n_dm, int nao,
                RysIntEnvVars envs, int *scheme, int *shls_slice,
                int ntile_ij_pairs, int ntile_kl_pairs,
                int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                ShellQuartet *pool, uint32_t *batch_head, int workers,
                int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = order / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    int lij = li + lj;
    int lkl = lk + ll;
    uint8_t stride_j = 1;
    uint8_t stride_k = lij + 1;
    uint8_t stride_l = lij + 1;
    int g_size = (lij + 1) * (lkl + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    JKMatrix jk = {vj, NULL, dm, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(uint32_t));

    if (!rys_j_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
#if CUDA_VERSION >= 12040
        gout_stride *= 2;
#endif
        int with_gout = scheme[2];
        dim3 threads(quartets_per_block, gout_stride);
        int nmax = MAX(lij, lkl);
        int nf3_ij = (lij+1)*(lij+2)*(lij+3)/6;
        int nf3_kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
        int buflen = (nroots*2 + g_size*3 + iprim*jprim + 9) * quartets_per_block;
        if (with_gout) {
            buflen += nf3_ij*nf3_kl * quartets_per_block;

            if (CHECK_SHARED_MEMORY_ATTRIBUTES) {
                cudaFuncAttributes attributes;
                const cudaError_t err_get_attribute = cudaFuncGetAttributes(&attributes, rys_j_with_gout_kernel);
                if (err_get_attribute != cudaSuccess) {
                    printf("Failed in cudaFuncGetAttributes(), attribute value is not reliable\n"); fflush(stdout);
                }
                if (buflen*sizeof(double) > attributes.maxDynamicSharedSizeBytes) {
                    printf("Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stdout);
                    fprintf(stderr, "Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stderr);
                }
            }

            rys_j_with_gout_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
        } else {
            buflen += (nf3_ij+nf3_kl*2+(lij+1)*(lkl+1)*(nmax+2)) * quartets_per_block;
            buflen += nf3_ij * TILE2; // dm_ij_cache

            if (CHECK_SHARED_MEMORY_ATTRIBUTES) {
                cudaFuncAttributes attributes;
                const cudaError_t err_get_attribute = cudaFuncGetAttributes(&attributes, rys_j_kernel);
                if (err_get_attribute != cudaSuccess) {
                    printf("Failed in cudaFuncGetAttributes(), attribute value is not reliable\n"); fflush(stdout);
                }
                if (buflen*sizeof(double) > attributes.maxDynamicSharedSizeBytes) {
                    printf("Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stdout);
                    fprintf(stderr, "Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stderr);
                }
            }

            rys_j_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        printf("CUDA Error in RYS_build_j, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stdout);
        fprintf(stderr, "CUDA Error in RYS_build_j, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_build_jk(double *vj, double *vk, double *dm, int n_dm, int nao,
                 RysIntEnvVars envs, int *scheme, int *shls_slice,
                 int ntile_ij_pairs, int ntile_kl_pairs,
                 int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                 float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                 ShellQuartet *pool, uint32_t *batch_head, int workers,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = order / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 1;
    uint8_t stride_k = stride_j * (lj + 1);
    uint8_t stride_l = stride_k * (lk + 1);
    uint16_t g_size = stride_l * (uint16_t)(ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    JKMatrix jk = {vj, vk, dm, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(uint32_t));

    if (order == 0) {
        os_jk_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers, omega);
    } else if (!rys_jk_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);

        const int j_cache_size = nfij + nfkl;
        const int k_cache_size = nfi * nfk + nfi * nfl + nfj * nfk + nfj * nfl;
        const int jk_cache_size = ((vj != NULL) ? j_cache_size : 0) + ((vk != NULL) ? k_cache_size : 0);
        const int root_g_size = nroots * 2 + g_size * 3;
        const int shared_root_g_jk_cache_size = (root_g_size > jk_cache_size) ? root_g_size : jk_cache_size;
        const int buflen = (9 + ij_prims + shared_root_g_jk_cache_size) * quartets_per_block;// + ij_prims*4*TILE2;

        if (CHECK_SHARED_MEMORY_ATTRIBUTES) {
            cudaFuncAttributes attributes;
            const cudaError_t err_get_attribute = cudaFuncGetAttributes(&attributes, rys_jk_kernel);
            if (err_get_attribute != cudaSuccess) {
                printf("Failed in cudaFuncGetAttributes(), attribute value is not reliable\n"); fflush(stdout);
            }
            if (buflen*sizeof(double) > attributes.maxDynamicSharedSizeBytes) {
                printf("Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stdout);
                fprintf(stderr, "Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stderr);
            }
        }

        rys_jk_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        printf("CUDA Error in RYS_build_jk, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stdout);
        fprintf(stderr, "CUDA Error in RYS_build_jk, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_build_jk_ip1(double *vj, double *vk, double *dm, int n_dm, int nao, int atom_offset,
                     RysIntEnvVars envs, int *scheme, int *shls_slice,
                     int ntile_ij_pairs, int ntile_kl_pairs,
                     int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                     float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                     ShellQuartet *pool, uint32_t *batch_head, int workers,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = (order + 1) / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 2;
    uint8_t stride_k = stride_j * (lj + 1);
    uint8_t stride_l = stride_k * (lk + 1);
    uint16_t g_size = stride_l * (uint16_t)(ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    JKMatrix jk = {vj, vk, dm, (uint16_t)n_dm, (uint16_t)atom_offset};
    cudaMemset(batch_head, 0, 2*sizeof(uint32_t));

    if (!rys_vjk_ip1_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*2 + g_size*3 + 6) * quartets_per_block;
        buflen += ij_prims*6;

        if (CHECK_SHARED_MEMORY_ATTRIBUTES) {
            cudaFuncAttributes attributes;
            const cudaError_t err_get_attribute = cudaFuncGetAttributes(&attributes, rys_jk_ip1_kernel);
            if (err_get_attribute != cudaSuccess) {
                printf("Failed in cudaFuncGetAttributes(), attribute value is not reliable\n"); fflush(stdout);
            }
            if (buflen*sizeof(double) > attributes.maxDynamicSharedSizeBytes) {
                printf("Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stdout);
                fprintf(stderr, "Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stderr);
            }
        }

        rys_jk_ip1_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        printf("CUDA Error in RYS_build_jk_ip1, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stdout);
        fprintf(stderr, "CUDA Error in RYS_build_jk_ip1, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_per_atom_jk_ip1(double *ejk, double j_factor, double k_factor,
                        double *dm, int n_dm, int nao,
                        RysIntEnvVars envs, int *scheme, int *shls_slice,
                        int ntile_ij_pairs, int ntile_kl_pairs,
                        int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                        float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                        ShellQuartet *pool, double *dd_pool, uint32_t *batch_head, int workers,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = (order + 1) / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 2;
    uint8_t stride_k = stride_j * (lj + 1);
    uint8_t stride_l = stride_k * (lk + 2);
    int g_size = stride_l * (uint16_t)(ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    if (n_dm == 1) { // RHF
        k_factor *= .5;
    }
    // *4 for the symmetry (i,j) = (j,i), (k,l) = (l,k) in J contraction
    // Additional factor 1/2 from the two-electron Coulomb operator
    JKEnergy jk = {ejk, dm, 2.*j_factor, -k_factor, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(int));

    if (!rys_ejk_ip1_unrolled(&envs, &jk, &bounds, pool, dd_pool, batch_head, scheme, workers)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*2 + g_size*3 + ij_prims + 9) * quartets_per_block;
        buflen = MAX(buflen, 12*gout_stride*quartets_per_block);

        if (CHECK_SHARED_MEMORY_ATTRIBUTES) {
            cudaFuncAttributes attributes;
            const cudaError_t err_get_attribute = cudaFuncGetAttributes(&attributes, rys_ejk_ip1_kernel);
            if (err_get_attribute != cudaSuccess) {
                printf("Failed in cudaFuncGetAttributes(), attribute value is not reliable\n"); fflush(stdout);
            }
            if (buflen*sizeof(double) > attributes.maxDynamicSharedSizeBytes) {
                printf("Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stdout);
                fprintf(stderr, "Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stderr);
            }
        }

        rys_ejk_ip1_kernel<<<workers, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, pool, dd_pool, batch_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        printf("CUDA Error in RYS_per_atom_jk_ip1, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stdout);
        fprintf(stderr, "CUDA Error in RYS_per_atom_jk_ip1, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_per_atom_jk_ip2_type12(double *ejk, double j_factor, double k_factor,
                        double *dm, int n_dm, int nao,
                        RysIntEnvVars envs, int *scheme, int *shls_slice,
                        int ntile_ij_pairs, int ntile_kl_pairs,
                        int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                        float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                        ShellQuartet *pool, double *dd_pool, uint32_t *batch_head, int workers,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = (order + 2) / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 2;
    uint8_t stride_k = stride_j * (lj + 2);
    uint8_t stride_l = stride_k * (lk + 2);
    int g_size = stride_l * (uint16_t)(ll + 2);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    if (n_dm > 1) { // UHF
        k_factor *= 2.;
    }
    // *4 for the symmetry (i,j) = (j,i), (k,l) = (l,k) in J contraction
    // Additional factor 1/2 from the two-electron Coulomb operator
    JKEnergy jk = {ejk, dm, 4.*j_factor, -k_factor, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(int));

    if (!rys_ejk_ip2_type12_unrolled(&envs, &jk, &bounds, pool, dd_pool, batch_head, scheme, workers)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*2 + g_size*3 + ij_prims + 9) * quartets_per_block;

        if (CHECK_SHARED_MEMORY_ATTRIBUTES) {
            cudaFuncAttributes attributes;
            const cudaError_t err_get_attribute = cudaFuncGetAttributes(&attributes, rys_ejk_ip2_type12_kernel);
            if (err_get_attribute != cudaSuccess) {
                printf("Failed in cudaFuncGetAttributes(), attribute value is not reliable\n"); fflush(stdout);
            }
            if (buflen*sizeof(double) > attributes.maxDynamicSharedSizeBytes) {
                printf("Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stdout);
                fprintf(stderr, "Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stderr);
            }
        }

        rys_ejk_ip2_type12_kernel<<<workers, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, pool, dd_pool, batch_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        printf("CUDA Error in RYS_per_atom_jk_ip2_type12, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stdout);
        fprintf(stderr, "CUDA Error in RYS_per_atom_jk_ip2_type12, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_per_atom_jk_ip2_type3(double *ejk, double j_factor, double k_factor,
                        double *dm, int n_dm, int nao,
                        RysIntEnvVars envs, int *scheme, int *shls_slice,
                        int ntile_ij_pairs, int ntile_kl_pairs,
                        int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                        float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
                        ShellQuartet *pool, double *dd_pool, uint32_t *batch_head, int workers,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint16_t ksh0 = shls_slice[4];
    uint16_t lsh0 = shls_slice[6];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    uint8_t ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    uint8_t lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfk = (lk+1)*(lk+2)/2;
    uint8_t nfl = (ll+1)*(ll+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t nfkl = nfk * nfl;
    uint8_t order = li + lj + lk + ll;
    uint8_t nroots = (order + 2) / 2 + 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) { // SR ERIs
        nroots *= 2;
    }
    uint8_t stride_j = li + 2;
    uint8_t stride_k = stride_j * (lj + 1);
    uint8_t stride_l = stride_k * (lk + 2);
    int g_size = stride_l * (uint16_t)(ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfk, nfij, nfkl,
        nroots, stride_j, stride_k, stride_l, iprim, jprim, kprim, lprim,
        ntile_ij_pairs, ntile_kl_pairs, tile_ij_mapping, tile_kl_mapping,
        q_cond, tile_q_cond, s_estimator, dm_cond, cutoff};

    if (n_dm > 1) { // UHF
        k_factor *= 2.;
    }
    // *4 for the symmetry (i,j) = (j,i), (k,l) = (l,k) in J contraction
    // Additional factor 1/2 from the two-electron Coulomb operator
    JKEnergy jk = {ejk, dm, 4.*j_factor, -k_factor, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(int));

    if (!rys_ejk_ip2_type3_unrolled(&envs, &jk, &bounds, pool, dd_pool, batch_head, scheme, workers)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*2 + g_size*3 + ij_prims + 9) * quartets_per_block;
        buflen = MAX(buflen, 9*gout_stride*quartets_per_block);

        if (CHECK_SHARED_MEMORY_ATTRIBUTES) {
            cudaFuncAttributes attributes;
            const cudaError_t err_get_attribute = cudaFuncGetAttributes(&attributes, rys_ejk_ip2_type3_kernel);
            if (err_get_attribute != cudaSuccess) {
                printf("Failed in cudaFuncGetAttributes(), attribute value is not reliable\n"); fflush(stdout);
            }
            if (buflen*sizeof(double) > attributes.maxDynamicSharedSizeBytes) {
                printf("Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stdout);
                fprintf(stderr, "Dynamic shared memory size in used (buflen*sizeof(double)) = %zu > set max value (attributes.maxDynamicSharedSizeBytes) = %zu\n", buflen*sizeof(double), attributes.maxDynamicSharedSizeBytes); fflush(stderr);
            }
        }

        rys_ejk_ip2_type3_kernel<<<workers, threads, buflen*sizeof(double)>>>(
                envs, jk, bounds, pool, dd_pool, batch_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        int device_id = -1;
        const cudaError_t err_get_device_id = cudaGetDevice(&device_id);
        if (err_get_device_id != cudaSuccess) {
            printf("Failed also in cudaGetDevice(), device_id value is not reliable\n"); fflush(stdout);
        }
        printf("CUDA Error in RYS_per_atom_jk_ip2_type3, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stdout);
        fprintf(stderr, "CUDA Error in RYS_per_atom_jk_ip2_type3, li,lj,lk,ll = %d,%d,%d,%d, device_id = %d, error message = %s\n", li,lj,lk,ll, device_id, cudaGetErrorString(err)); fflush(stderr);
        return 1;
    }
    return 0;
}

int RYS_init_constant(int *g_pair_idx, int *offsets,
                      double *env, int env_size, int shm_size)
{
    // TODO: test whether the constant memory c_env can improve performance
    //cudaMemcpyToSymbol(c_env, env, sizeof(double)*env_size);
    cudaMemcpyToSymbol(c_g_pair_idx, g_pair_idx, 3675*sizeof(int));
    cudaMemcpyToSymbol(c_g_pair_offsets, offsets, sizeof(int) * LMAX1*LMAX1);
    cudaFuncSetAttribute(rys_jk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_jk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_ejk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_ejk_ip2_type12_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_ejk_ip2_type3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int RYS_init_rysj_constant(int shm_size)
{
    Fold2Index i_in_fold2idx[165];
    Fold3Index i_in_fold3idx[495];
    int n2 = 0;
    int n3 = 0;
    for (int l = 0; l <= LMAX*2; ++l) {
        for (int i = 0, ijk = 0; i <= l; ++i) {
        for (int j = 0; j <= l-i; ++j, ++n2) {
            i_in_fold2idx[n2].x = i;
            i_in_fold2idx[n2].y = j;
            i_in_fold2idx[n2].fold3offset = ijk;
            for (int k = 0; k <= l-i-j; ++k, ++n3, ++ijk) {
                i_in_fold3idx[n3].x = i;
                i_in_fold3idx[n3].y = j;
                i_in_fold3idx[n3].z = k;
                i_in_fold3idx[n3].fold2yz = (l+1)*(l+2)/2 - (l-j+1)*(l-j+2)/2 + k;
            }
        } }
    }
    cudaMemcpyToSymbol(c_i_in_fold2idx, i_in_fold2idx, 165*sizeof(Fold2Index));
    cudaMemcpyToSymbol(c_i_in_fold3idx, i_in_fold3idx, 495*sizeof(Fold3Index));
    cudaFuncSetAttribute(rys_j_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_j_with_gout_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int cuda_version()
{
    return CUDA_VERSION;
}
}
