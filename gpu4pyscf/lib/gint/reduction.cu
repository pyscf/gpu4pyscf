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

#pragma once

template <int blockx, int blocky>
__device__ static void block_reduce_x(double val, double *addr, int tx, int ty){
    __shared__ double sdata[blockx*blocky];
    sdata[tx*blocky+ty] = val; __syncthreads();
    if (blockx >= 32) if (tx < 16) sdata[tx*blocky+ty] += sdata[(tx+16)*blocky+ty]; __syncthreads();
    if (blockx >= 16) if (tx < 8)  sdata[tx*blocky+ty] += sdata[(tx+8)*blocky+ty];  __syncthreads();
    if (blockx >= 8)  if (tx < 4)  sdata[tx*blocky+ty] += sdata[(tx+4)*blocky+ty];  __syncthreads();
    if (blockx >= 4)  if (tx < 2)  sdata[tx*blocky+ty] += sdata[(tx+2)*blocky+ty];  __syncthreads();
    if (blockx >= 2)  if (tx < 1)  sdata[tx*blocky+ty] += sdata[(tx+1)*blocky+ty];  __syncthreads();
    if (tx == 0) atomicAdd(addr, sdata[ty]);
}

template <int blockx, int blocky>
__device__ static void block_reduce_y(double val, double *addr, int tx, int ty){
    /*
    if(blocky >= 32) sdata[tx*blocky+ty] += sdata[tx*blocky+ty+16];
    if(blocky >= 16) sdata[tx*blocky+ty] += sdata[tx*blocky+ty+8];
    if(blocky >= 8)  sdata[tx*blocky+ty] += sdata[tx*blocky+ty+4];
    if(blocky >= 4)  sdata[tx*blocky+ty] += sdata[tx*blocky+ty+2];
    if(blocky >= 2)  sdata[tx*blocky+ty] += sdata[tx*blocky+ty+1];
    */
    int stride = blocky + 1;
    __shared__ double sdata[blockx*(blocky+1)];
    sdata[tx*stride+ty] = val; __syncthreads();
    if (blocky >= 32) if (ty < 16) sdata[tx*stride+ty] += sdata[tx*stride+ty+16]; __syncthreads();
    if (blocky >= 16) if (ty < 8)  sdata[tx*stride+ty] += sdata[tx*stride+ty+8];  __syncthreads();
    if (blocky >= 8)  if (ty < 4)  sdata[tx*stride+ty] += sdata[tx*stride+ty+4];  __syncthreads();
    if (blocky >= 4)  if (ty < 2)  sdata[tx*stride+ty] += sdata[tx*stride+ty+2];  __syncthreads();
    if (blocky >= 2)  if (ty < 1)  sdata[tx*stride+ty] += sdata[tx*stride+ty+1];  __syncthreads();
    if (ty == 0) atomicAdd(addr, sdata[tx*stride]);
 }

template <int BLKSIZE> 
__device__ void block_reduce(double *sum, double a){
    const int tx = threadIdx.x;
    __syncthreads();
    __shared__ double as[BLKSIZE];
    as[tx] = a;
    __syncthreads();

    if (BLKSIZE >= 512 && tx < 256) as[tx] += as[tx + 256]; __syncthreads();
    if (BLKSIZE >= 256 && tx < 128) as[tx] += as[tx + 128]; __syncthreads();
    if (BLKSIZE >= 128 && tx < 64)  as[tx] += as[tx + 64]; __syncthreads();
    if (tx < 32){
        volatile double* vs = as;
        vs[tx] += vs[tx+32];
        vs[tx] += vs[tx+16];
        vs[tx] += vs[tx+8];
        vs[tx] += vs[tx+4];
        vs[tx] += vs[tx+2];
        vs[tx] += vs[tx+1];
    }
    if (tx == 0){
        atomicAdd(sum, as[0]);
    }
}
