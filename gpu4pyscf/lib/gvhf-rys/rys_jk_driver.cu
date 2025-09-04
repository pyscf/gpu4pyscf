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

extern "C" {
int RYS_init_constant(int *g_pair_idx, int *offsets,
                      double *env, int env_size, int shm_size)
{
    // TODO: test whether the constant memory c_env can improve performance
    //cudaMemcpyToSymbol(c_env, env, sizeof(double)*env_size);
    cudaMemcpyToSymbol(c_g_pair_idx, g_pair_idx, 3675*sizeof(int));
    cudaMemcpyToSymbol(c_g_pair_offsets, offsets, sizeof(int) * LMAX1*LMAX1);
    return 0;
}

int cuda_version()
{
    return CUDA_VERSION;
}
}
