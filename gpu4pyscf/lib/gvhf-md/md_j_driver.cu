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
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/md_j.cuh"

#define RT2_MAX 9

int offset_for_Rt2_idx(int lij, int lkl)
{
    return host_Rt2_idx_offsets[lij*RT2_MAX+lkl];
}

int qd_offset_for_threads(int npairs, int threads)
{
    int npairs_aligned = (npairs + 31) & 0xffffffe0; // 32-element aligned
    int address = 0;
    for (int i = 1; i < threads; i *= 2) {
        address += npairs_aligned;
        npairs_aligned /= 2;
    }
    return address;
}

void initialize_Rt2_indices();

extern __global__
void md_j_1dm_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                     int threadsx, int threadsy, int tilex, int tiley,
                     uint16_t *pRt2_kl_ij, int8_t *efg_phase);
extern __global__
void md_j_4dm_kernel(RysIntEnvVars envs, JKMatrix jk, MDBoundsInfo bounds,
                     int threadsx, int threadsy, int tilex, int tiley, int dm_size,
                     uint16_t *pRt2_kl_ij, int8_t *efg_phase);
extern __global__
void pbc_md_j_kernel(RysIntEnvVars envs, JKMatrix jmat, MDBoundsInfo bounds,
                  int threadsx, int threadsy, int tilex, int tiley,
                  uint16_t *pRt2_kl_ij, int8_t *efg_phase);

extern "C" {
int init_mdj_constant(int shm_size)
{
    initialize_Rt2_indices();
    cudaFuncSetAttribute(md_j_1dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(md_j_4dm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaFuncSetAttribute(pbc_md_j_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
