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

__device__ int CART_QUART_OFFSETS[LMAX1*LMAX1*LMAX1*LMAX1];
__device__ uchar4 CART_QUART_INDICES[35*35*35*35];

extern "C" {
int RYS_init_constant()
{
    int *offsets = (int *)malloc(sizeof(int) * LMAX1*LMAX1*LMAX1*LMAX1);
    uint8_t *indices = (uint8_t *)malloc(sizeof(uint8_t) * 4 * 35*35*35*35);
    int offset = 0;
    for (int ll = 0; ll <= LMAX; ++ll) {
    for (int lk = 0; lk <= LMAX; ++lk) {
    for (int lj = 0; lj <= LMAX; ++lj) {
    for (int li = 0; li <= LMAX; ++li) {
        offsets[((ll*LMAX1+lk)*LMAX1+lj)*LMAX1+li] = offset;
        for (int l = 0; l <= ll; ++l) {
        for (int k = 0; k <= lk; ++k) {
        for (int j = 0; j <= lj; ++j) {
        for (int i = 0; i <= li; ++i) {
            indices[offset*4+0] = i;
            indices[offset*4+1] = j;
            indices[offset*4+2] = k;
            indices[offset*4+3] = l;
            offset++;
        } } } }
    } } } }
    cudaMemcpy(CART_QUART_OFFSETS, offsets, sizeof(int) * LMAX1*LMAX1*LMAX1*LMAX1, cudaMemcpyHostToDevice);
    cudaMemcpy(CART_QUART_INDICES, indices, sizeof(uint8_t) * 4 * 35*35*35*35, cudaMemcpyHostToDevice);
    free(indices);
    free(offsets);
    return 0;
}

int cuda_version()
{
    return CUDA_VERSION;
}
}
