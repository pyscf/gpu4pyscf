#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define BLOCK_SIZE      16

__global__ static
void fill_indexed_triu_kernel(double *out, int *tril_idx, int *ki_idx,
                              int npairs, int nao, int naux)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockIdx_x = item.get_group(1);
    int blockIdx_y = item.get_group(0);
    int blockDim_x = item.get_local_range(1);
    #else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int blockDim_x = blockDim.x;
    #endif
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
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int threadIdx_x = item.get_local_id(0);
    int blockIdx_x = item.get_group(0);
    int blockDim_x = item.get_local_range(0);
    #else
    int threadIdx_x = threadIdx.x;
    int blockIdx_x = blockIdx.x;
    int blockDim_x = blockDim.x;
    #endif
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
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<1>();
    int pair_id = item.get_global_id(1);
    #else
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    #endif
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
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int j = item.get_global_id(1);
    int i = item.get_global_id(0);
    #else
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    #endif
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
    #ifdef USE_SYCL
    sycl::range<2> threads(BLOCK_SIZE, 32);
    sycl::range<2> blocks(nkpts, (npairs+BLOCK_SIZE-1)/BLOCK_SIZE);
    sycl_get_queue()->parallel_for<class fill_indexed_triu_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      fill_indexed_triu_kernel(out, tril_idx, ki_idx, npairs, nao, naux);
    });
    #else
    dim3 threads(32, BLOCK_SIZE);
    dim3 blocks((npairs+BLOCK_SIZE-1)/BLOCK_SIZE, nkpts);
    fill_indexed_triu_kernel<<<blocks, threads>>>(
        out, tril_idx, ki_idx, npairs, nao, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_indexed_triu: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int fill_bvk_triu(double *out, int *pair_address, int *conj_mapping,
                  int npairs, int bvk_ncells, int nao, int naux)
{
    #ifdef USE_SYCL
    if (naux == 1) {
        int blocks = (npairs+255)/256;
        sycl_get_queue()->parallel_for<class fill_bvk_triu_axis0_sycl>(sycl::nd_range<1>(blocks * 256, 256), [=](auto item) {
          fill_bvk_triu_naux1_kernel(out, pair_address, conj_mapping, npairs, bvk_ncells, nao);
        });
    } else {
        sycl_get_queue()->parallel_for<class fill_bvk_triu_sycl>(sycl::nd_range<1>(npairs * 256, 256), [=](auto item) {
          fill_bvk_triu_kernel(out, pair_address, conj_mapping, bvk_ncells, nao, naux);
        });
    }
    #else
    if (naux == 1) {
        dim3 blocks((npairs+255)/256);
        fill_bvk_triu_naux1_kernel<<<blocks, 256>>>(
            out, pair_address, conj_mapping, npairs, bvk_ncells, nao);
    } else {
        fill_bvk_triu_kernel<<<npairs, 256>>>(
            out, pair_address, conj_mapping, bvk_ncells, nao, naux);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_bvk_triu: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int fill_bvk_triu_axis0(double *out, int *conj_mapping, int nao, int bvk_ncells)
{
    int nao_b = (nao + BLOCK_SIZE-1) / BLOCK_SIZE;
    #ifdef USE_SYCL
    sycl::range<2> threads(BLOCK_SIZE, BLOCK_SIZE);
    sycl::range<2> blocks(nao_b, nao_b);
    sycl_get_queue()->parallel_for<class fill_bvk_triu_axis0_sycl2>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      fill_bvk_triu_axis0_kernel(out, conj_mapping, bvk_ncells, nao);
    });
    #else
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(nao_b, nao_b);
    fill_bvk_triu_axis0_kernel<<<blocks, threads>>>(out, conj_mapping, bvk_ncells, nao);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in fill_bvk_triu_axis0: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
