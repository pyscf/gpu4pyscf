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

#include "gvhf-rys/vhf.cuh"

#define BLOCK_SIZE      128

__global__ static
void filter_q_cond_by_distance_kernel(float *q_cond, float *s_estimator, RysIntEnvVars envs,
                                      float *atom_diffuse_exps, float *s_max_per_atom,
                                      float log_cutoff, int natm_cell0
                                      #ifdef USE_SYCL
                                      , sycl::nd_item<2> &item, char *shm_mem
                                      #endif
                                      )
{
#ifdef USE_SYCL
    int threadIdx_x = item.get_local_id(1);
    int threadIdx_y = item.get_local_id(0);
    int blockIdx_x = item.get_group(1);
    int blockIdx_y = item.get_group(0);
    int blockDim_x = item.get_local_range(1);
    int blockDim_y = item.get_local_range(0);
    double *xyz_cache = reinterpret_cast<double *>(shm_mem);
#else
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    extern __shared__ float xyz_cache[];    
#endif
    if (blockIdx_y < blockIdx_x) { // i < j
        return;
    }
    int tx = threadIdx_x;
    int ty = threadIdx_y;
    int threads = blockDim_x * blockDim_y;
    int thread_id = tx + blockDim_x * ty;
    uint32_t nbas = envs.nbas;
    int ish0 = blockIdx_y * BLOCK_SIZE + ty;
    int jsh0 = blockIdx_x * BLOCK_SIZE + tx;
    int ish1 = min(ish0 + BLOCK_SIZE, static_cast<int>(nbas));
    int jsh1 = min(jsh0 + BLOCK_SIZE, static_cast<int>(nbas));
    jsh1 = min(ish1, jsh1);

    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
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
    for (int ish = ish0; ish < ish1; ish += blockDim_y) {
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        float ai = diffuse_exps[ish];
        float xi = ri[0];
        float yi = ri[1];
        float zi = ri[2];
        for (int jsh = jsh0; jsh < min(ish+1, jsh1); jsh += blockDim_x) {
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
    int buflen = natm_cell0 * 3 * sizeof(float);

    #ifdef USE_SYCL
    sycl::range<2> threads(16, 16);
    sycl::range<2> blocks(sh_blocks, sh_blocks);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(buflen), cgh);
      cgh.parallel_for<class filter_q_cond_by_distance_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      filter_q_cond_by_distance_kernel(q_cond, s_estimator, dev_envs, diffuse_exps_per_atom, s_max_per_atom,
                                       log_cutoff, natm_cell0,
                                       item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    dim3 threads(16, 16);
    dim3 blocks(sh_blocks, sh_blocks);
    filter_q_cond_by_distance_kernel<<<blocks, threads, buflen>>>(
        q_cond, s_estimator, *envs, diffuse_exps_per_atom, s_max_per_atom,
        log_cutoff, natm_cell0);
    #endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in filter_q_cond_by_distance error message = %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
