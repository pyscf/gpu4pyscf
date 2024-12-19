#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gvhf-rys/vhf.cuh"
#include "ft_ao.h"

__constant__ int c_g_pair_idx[3675];
__constant__ int c_g_pair_offsets[LMAX1*LMAX1];

extern __global__
void ft_aopair_kernel(double *out, AFTIntEnvVars envs, AFTBoundsInfo bounds);
extern __global__
void ft_aopair_fill_triu(double *out, int *conj_mapping, int bvk_ncells, int nGv);

int ft_ao_unrolled(double *out, AFTIntEnvVars *envs, AFTBoundsInfo *bounds, int *scheme);

extern "C" {
int PBC_build_ft_ao(double *out, AFTIntEnvVars *envs,
                    int *scheme, int *shls_slice, int npairs_ij, int ngrids,
                    int *ish_in_pair, int *jsh_in_pair, double *grids,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
    uint16_t ish0 = shls_slice[0];
    uint16_t jsh0 = shls_slice[2];
    uint8_t li = bas[ANG_OF + ish0*BAS_SLOTS];
    uint8_t lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    uint8_t iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    uint8_t jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    uint8_t nfi = (li+1)*(li+2)/2;
    uint8_t nfj = (lj+1)*(lj+2)/2;
    uint8_t nfij = nfi * nfj;
    uint8_t stride_i = 1;
    uint8_t stride_j = li + 1;
    // up to g functions
    uint8_t g_size = stride_j * (uint16_t)(lj + 1);
    AFTBoundsInfo bounds = {li, lj, nfij, g_size,
        stride_i, stride_j, iprim, jprim,
        npairs_ij, ish_in_pair, jsh_in_pair, ngrids, grids};

    if (!ft_ao_unrolled(out, envs, &bounds, scheme)) {
        int nGv_per_block = scheme[0];
        int gout_stride = scheme[1];
        int nsp_per_block = scheme[2];
        dim3 threads(nGv_per_block, gout_stride, nsp_per_block);
        int sp_blocks = (npairs_ij + nsp_per_block - 1) / nsp_per_block;
        int Gv_batches = (ngrids + nGv_per_block - 1) / nGv_per_block;
        dim3 blocks(sp_blocks, Gv_batches);
        int buflen = g_size*6 * nGv_per_block * nsp_per_block;
        ft_aopair_kernel<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, bounds);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBC_build_ft_ao: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBC_ft_aopair_fill_triu(double *out, int *conj_mapping, int nao, int bvk_ncells, int nGv)
{
    int nGv2 = nGv * 2; // *2 for complex number
    int threads = 1024;
    dim3 blocks(nao, nao);
    ft_aopair_fill_triu<<<blocks, threads>>>(out, conj_mapping, bvk_ncells, nGv2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in PBC_ft_aopair_fill_triu: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBC_FT_init_constant(int *g_pair_idx, int *offsets,
                         double *env, int env_size, int shm_size)
{
    cudaMemcpyToSymbol(c_g_pair_idx, g_pair_idx, 3675*sizeof(int));
    cudaMemcpyToSymbol(c_g_pair_offsets, offsets, sizeof(int) * LMAX1*LMAX1);
    cudaFuncSetAttribute(ft_aopair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
