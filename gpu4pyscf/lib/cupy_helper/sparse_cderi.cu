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
#include "gint/cuda_alloc.cuh"
#define THREADS        32

typedef struct {
    int naux;
    int nij;
    long *row;
    long *col;
    double *data;
}CDERI_BLOCK;

typedef struct {
    int nblocks;
    int nblocks_max;
    int nao;
    CDERI_BLOCK *blocks;
}CDERI;

__global__
void _unpack(CDERI_BLOCK block, int nao, int offset, double *out){
    int ij = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int nij = block.nij;

    int idx_aux = k + offset;
    if (idx_aux >= block.naux || ij >= nij){
        return;
    }
    int i = block.row[ij];
    int j = block.col[ij];

    double e = block.data[idx_aux * nij + ij];
    out[k * nao * nao + i * nao + j] = e;
    out[k * nao * nao + j * nao + i] = e;
}


extern "C" {__host__

void init_cderi(CDERI **pcderi, int nblocks_max, int nao){
    CDERI *cderi = (CDERI *)malloc(sizeof(CDERI));
    memset(cderi, 0, sizeof(CDERI));
    cderi->nao = nao;
    cderi->nblocks = 0;
    cderi->nblocks_max = nblocks_max;
    cderi->blocks = (CDERI_BLOCK *)malloc(sizeof(CDERI_BLOCK) * nblocks_max);
    *pcderi = cderi;
}

int add_block(CDERI **pcderi, int nij, int naux, long *row, long *col, double *data){
    CDERI *cderi = *pcderi;
    CDERI_BLOCK *block = cderi->blocks + cderi->nblocks;
    block->nij = nij;
    block->row = row;
    block->col = col;
    block->data = data;
    block->naux = naux;
    cderi->nblocks += 1;
    return 0;
}

void delete_cderi(CDERI **pcderi){
    CDERI *cderi = *pcderi;
    /*
    for (int i = 0; i < cderi->nblocks; i++){
        CDERI_BLOCK *block = cderi->blocks + i;
        FREE(block->row);
        FREE(block->col);
        FREE(block->data);
    }
    */
    free(cderi->blocks);
    free(cderi);
    pcderi = NULL;
}

int unpack_block(CDERI_BLOCK *block, int p1, int p2, int nao, double *buf){
    int nij = block->nij;
    int blockx = (nij + THREADS - 1) / THREADS;
    int blocky = (p2 - p1 + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(blockx, blocky);

    _unpack<<<blocks, threads>>>(*block, nao, p1, buf);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int unpack(CDERI **pcderi, int p1, int p2, double *buf){
    CDERI *cderi = *pcderi;
    int nao = cderi->nao;
    for (int i = 0; i < cderi->nblocks; i++){
        CDERI_BLOCK *block = cderi->blocks + i;
        int err = unpack_block(block, p1, p2, nao, buf);
        if(err != 0){
            return err;
        }
    }
    return 0;
}

}
