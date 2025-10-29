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
static int _Rt2_idx_offsets[] = {
0,1,5,15,35,70,126,210,330,
495,499,515,555,635,775,999,1335,1815,
2475,2485,2525,2625,2825,3175,3735,4575,5775,
7425,7445,7525,7725,8125,8825,9945,11625,14025,
17325,17360,17500,17850,18550,19775,21735,24675,28875,
34650,34706,34930,35490,36610,38570,41706,46410,53130,
62370,62454,62790,63630,65310,68250,72954,80010,90090,
103950,104070,104550,105750,108150,112350,119070,129150,143550,
163350,163515,164175,165825,169125,174900,184140,198000,217800,
245025,
};

int offset_for_Rt2_idx(int lij, int lkl)
{
    return _Rt2_idx_offsets[lij*RT2_MAX+lkl];
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
