/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
