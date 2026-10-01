#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gsycl/gpu_compat.h"

#define BLOCK_SIZE      16

__global__ static
void fill_indexed_triu_kernel(double *out, int *tril_idx, int *ki_idx,
                              int npairs, int nao, int naux)
{
    setup_context();
    int pair_id = blockIdx_x * BLOCK_SIZE + threadIdx_y;
    if (pair_id >= npairs) {
        return;
    }
    int pair_ij = tril_idx[pair_id];
    int kp = blockIdx_y;
    size_t Nao = nao;
    size_t Naux = naux;
    int ij = pair_ij + kp * Nao * Nao;
    int i = pair_ij / nao;
    int j = pair_ij - nao * i;
    int ki = ki_idx[kp];
    int ji = (ki * nao + j) * Nao + i;
    if (ji == ij) return;

    for (int aux_id = threadIdx_x; aux_id < naux; aux_id += blockDim_x) {
        out[ji*Naux+aux_id] = out[ij*Naux+aux_id];
    }
}

__global__ static
void fill_bvk_triu_kernel(double *out, int *pair_address, int *conj_mapping,
                          int bvk_ncells, int nao, int naux)
{
    setup_context();
    int ij = pair_address[blockIdx_x];
    int r = ij / nao;
    int j = ij - nao * r;
    int i = r / bvk_ncells;
    int cell_j = r - bvk_ncells * i;
    int cell_conj = conj_mapping[cell_j];
    int ji = j * (bvk_ncells * nao) + cell_conj * nao + i;
    if (ji == ij) return;

    size_t Naux = naux;
    for (int aux_id = threadIdx_x; aux_id < naux; aux_id += blockDim_x) {
        out[ji*Naux+aux_id] = out[ij*Naux+aux_id];
    }
}

__global__ static
void fill_bvk_triu_naux1_kernel(double *out, int *pair_address, int *conj_mapping,
                                int npairs, int bvk_ncells, int nao)
{
    setup_context();
    int pair_id = global_x;
    if (pair_id >= npairs) return;
    int ij = pair_address[pair_id];
    int r = ij / nao;
    int j = ij - nao * r;
    int i = r / bvk_ncells;
    int cell_j = r - bvk_ncells * i;
    int cell_conj = conj_mapping[cell_j];
    int ji = j * (bvk_ncells * nao) + cell_conj * nao + i;
    if (ji == ij) return;
    out[ji] = out[ij];
}

__global__ static
void fill_bvk_triu_axis0_kernel(double *out, int *conj_mapping, int bvk_ncells, int nao)
{
    setup_context();
    int j = global_x;
    int i = global_y;
    if (i >= nao || j >= nao || i <= j) {
        return;
    }
    size_t nao2 = nao * nao;
    size_t ij = i * nao + j;
    size_t ji = j * nao + i;
    for (int k = 0; k < bvk_ncells; ++k) {
        int ck = conj_mapping[k];
        out[ji + ck*nao2] = out[ij + k*nao2];
    }
}

extern "C" {
int fill_indexed_triu(double *out, int *tril_idx, int *ki_idx,
                      int npairs, int nkpts, int nao, int naux)
{
    auto threads = make_block(32, BLOCK_SIZE);
    auto blocks = make_grid((npairs+BLOCK_SIZE-1)/BLOCK_SIZE, nkpts);
    LAUNCH_KERNEL(fill_indexed_triu_kernel, blocks, threads, 0,
        out, tril_idx, ki_idx, npairs, nao, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_indexed_triu: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int fill_bvk_triu(double *out, int *pair_address, int *conj_mapping,
                  int npairs, int bvk_ncells, int nao, int naux)
{
    auto threads = make_block(256);
    if (naux == 1) {
        auto blocks = make_grid((npairs+255)/256);
        LAUNCH_KERNEL(fill_bvk_triu_naux1_kernel, blocks, threads, 0,
            out, pair_address, conj_mapping, npairs, bvk_ncells, nao);
    } else {
        auto blocks = make_grid(npairs);
        LAUNCH_KERNEL(fill_bvk_triu_kernel, blocks, threads, 0,
            out, pair_address, conj_mapping, bvk_ncells, nao, naux);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_bvk_triu: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int fill_bvk_triu_axis0(double *out, int *conj_mapping, int nao, int bvk_ncells)
{
    int nao_b = (nao + BLOCK_SIZE-1) / BLOCK_SIZE;
    auto threads = make_block(BLOCK_SIZE, BLOCK_SIZE);
    auto blocks = make_grid(nao_b, nao_b);
    LAUNCH_KERNEL(fill_bvk_triu_axis0_kernel, blocks, threads, 0,
        out, conj_mapping, bvk_ncells, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_bvk_triu_axis0: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
