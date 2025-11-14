/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_runtime.h>

#include "gvhf-rys/vhf.cuh"

#define BLOCK_SIZE      128

__global__ static
void filter_q_cond_by_distance_kernel(float *q_cond, float *s_estimator, RysIntEnvVars envs,
                                      float *atom_diffuse_exps, float *s_max_per_atom,
                                      float log_cutoff, int natm_cell0)
{
    if (blockIdx.y < blockIdx.x) { // i < j
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    int thread_id = tx + blockDim.x * ty;
    uint32_t nbas = envs.nbas;
    int ish0 = blockIdx.y * BLOCK_SIZE + ty;
    int jsh0 = blockIdx.x * BLOCK_SIZE + tx;
    int ish1 = min(ish0 + BLOCK_SIZE, nbas);
    int jsh1 = min(jsh0 + BLOCK_SIZE, nbas);
    jsh1 = min(ish1, jsh1);

    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ float xyz_cache[];
    for (int k = thread_id; k < natm_cell0; k += threads) {
        double *rk = env + atm[k*ATM_SLOTS+PTR_COORD];
        xyz_cache[k*3+0] = rk[0];
        xyz_cache[k*3+1] = rk[1];
        xyz_cache[k*3+2] = rk[2];
    }

    float omega = env[PTR_RANGE_OMEGA];
    if (omega == 0) {
        omega = 0.1f;
    }
    float omega2 = omega * omega;
    float *diffuse_exps = s_estimator + nbas*nbas;
    for (int ish = ish0; ish < ish1; ish += blockDim.y) {
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        float ai = diffuse_exps[ish];
        float xi = ri[0];
        float yi = ri[1];
        float zi = ri[2];
        for (int jsh = jsh0; jsh < min(ish+1, jsh1); jsh += blockDim.x) {
            uint32_t bas_ij = ish * nbas + jsh;
            if (q_cond[bas_ij] < log_cutoff-8.f) {
                continue;
            }
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            float aj = diffuse_exps[jsh];
            float aij = ai + aj;
            float aj_aij = aj / aij;
            float theta = (omega2 * aij) / (omega2 + aij);
            float xj = rj[0];
            float yj = rj[1];
            float zj = rj[2];
            float xjxi = xj - xi;
            float yjyi = yj - yi;
            float zjzi = zj - zi;
            float xpa = xjxi * aj_aij;
            float ypa = yjyi * aj_aij;
            float zpa = zjzi * aj_aij;
            float xij = xi + xpa;
            float yij = yi + ypa;
            float zij = zi + zpa;
            float s_ij = s_estimator[bas_ij];
            float rr_cutoff = s_ij - log_cutoff;  
            int negligible = 1;
            for (int k = 0; k < natm_cell0; ++k) {
                float dx = xij - xyz_cache[k*3+0];
                float dy = yij - xyz_cache[k*3+1];
                float dz = zij - xyz_cache[k*3+2];
                float rr = dx * dx + dy * dy + dz * dz;
                float ak = atom_diffuse_exps[k]*2;
                float s_kl_guess = s_max_per_atom[k]; // from s_estimator diagonal
                float theta_k = theta * ak / (theta + ak);
                float theta_rr = theta_k * rr;
                if (theta_rr - s_kl_guess < rr_cutoff) {
                    negligible = 0;
                    break;
                }
            }
            if (negligible) {
                q_cond[bas_ij] = -500.f;
                q_cond[jsh*nbas+ish] = -500.f;
            }
        }
    }
}

extern "C" {
int filter_q_cond_by_distance(float *q_cond, float *s_estimator, RysIntEnvVars *envs,
                              float *diffuse_exps_per_atom, float *s_max_per_atom,
                              float log_cutoff, int natm_cell0, int nbas)
{
    int sh_blocks = (nbas + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads(16, 16);
    dim3 blocks(sh_blocks, sh_blocks);
    int buflen = natm_cell0 * 3 * sizeof(float);
    filter_q_cond_by_distance_kernel<<<blocks, threads, buflen>>>(
        q_cond, s_estimator, *envs, diffuse_exps_per_atom, s_max_per_atom,
        log_cutoff, natm_cell0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in filter_q_cond_by_distance error message = %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
