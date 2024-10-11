#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "vhf.cuh"

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
extern __global__ void rys_sr_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                     ShellQuartet *pool, uint32_t *batch_head);
extern __global__ void rys_jk_ip1_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                         ShellQuartet *pool, uint32_t *batch_head);
extern int rys_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head,
                    int *scheme, int workers, double omega);
extern int rys_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head,
                    int *scheme, int workers, double omega);
extern int rys_sr_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head,
                    int *scheme, int workers, double omega);
extern int os_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head,
                    int *scheme, int workers, double omega);

extern "C" {
int RYS_build_j(double *vj, double *dm, int n_dm, int nao,
                RysIntEnvVars envs, int *scheme, int *shls_slice,
                int ntile_ij_pairs, int ntile_kl_pairs,
                int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                float *q_cond, float *dm_cond, float cutoff,
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
        q_cond, dm_cond, cutoff};

    JKMatrix jk = {vj, NULL, dm, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(uint32_t));

    if (!rys_j_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers, omega)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int with_gout = scheme[2];
        dim3 threads(quartets_per_block, gout_stride);
        int nmax = MAX(lij, lkl);
        int nf3_ij = (lij+1)*(lij+2)*(lij+3)/6;
        int nf3_kl = (lkl+1)*(lkl+2)*(lkl+3)/6;
        int buflen = (nroots*2 + g_size*3 + iprim*jprim*4) * quartets_per_block;
        if (with_gout) {
            buflen += nf3_ij*nf3_kl * quartets_per_block;
            rys_j_with_gout_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
        } else {
            buflen += (nf3_ij+nf3_kl*2+(lij+1)*(lkl+1)*(nmax+2)) * quartets_per_block;
            buflen += nf3_ij * TILE2; // dm_ij_cache
            rys_j_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in RYS_build_j: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int RYS_build_jk(double *vj, double *vk, double *dm, int n_dm, int nao,
                 RysIntEnvVars envs, int *scheme, int *shls_slice,
                 int ntile_ij_pairs, int ntile_kl_pairs,
                 int *tile_ij_mapping, int *tile_kl_mapping, float *tile_q_cond,
                 float *q_cond, float *dm_cond, float cutoff,
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
        q_cond, dm_cond, cutoff};

    JKMatrix jk = {vj, vk, dm, (uint16_t)n_dm};
    cudaMemset(batch_head, 0, 2*sizeof(uint32_t));

    if (omega >= 0) {
        if (order <= 0) {
            os_jk_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers, omega);
        } else if (!rys_jk_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers, omega)) {
            int quartets_per_block = scheme[0];
            int gout_stride = scheme[1];
            int ij_prims = iprim * jprim;
            dim3 threads(quartets_per_block, gout_stride);
            int buflen = (nroots*2 + g_size*3 + ij_prims*4) * quartets_per_block;// + ij_prims*4*TILE2;
            rys_jk_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
        }
    } else if (!rys_sr_jk_unrolled(&envs, &jk, &bounds, pool, batch_head, scheme, workers, omega)) {
        int quartets_per_block = scheme[0];
        int gout_stride = scheme[1];
        int ij_prims = iprim * jprim;
        dim3 threads(quartets_per_block, gout_stride);
        int buflen = (nroots*4 + g_size*3 + ij_prims*4) * quartets_per_block;// + ij_prims*4*TILE2;
        rys_sr_jk_kernel<<<workers, threads, buflen*sizeof(double)>>>(envs, jk, bounds, pool, batch_head);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in RYS_build_jk: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

void RYS_init_constant(int *g_pair_idx, int *offsets,
                       double *env, int env_size, int shm_size)
{
    // TODO: test whether the constant memory c_env can improve performance
    //cudaMemcpyToSymbol(c_env, env, sizeof(double)*env_size);
    cudaMemcpyToSymbol(c_g_pair_idx, g_pair_idx, 3675*sizeof(int));
    cudaMemcpyToSymbol(c_g_pair_offsets, offsets, sizeof(int) * LMAX1*LMAX1);
    cudaFuncSetAttribute(rys_jk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(rys_sr_jk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
}

void RYS_init_rysj_constant(int shm_size)
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
}

int cuda_version()
{
    return CUDA_VERSION;
}
}
