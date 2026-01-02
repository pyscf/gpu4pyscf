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
#include <string.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

#define THREADS         256
#define WARP_SIZE       32
#define WARPS           8
#define REMOTE_THRESHOLD 50

__device__ __forceinline__
void _filter_jk_images(int& img_counts, uint32_t *img_pool, PBCIntEnvVars &envs,
                       uint32_t bas_ij, int kidx0, int kidx1, int li, int lj,
                       int *ksh_idx, int *img_idx, uint32_t *sp_img_offsets,
                       float *diffuse_exps, float *diffuse_coefs,
                       float *atom_aux_exps, float log_cutoff)
{
    int thread_id = threadIdx.x;
    int lane = thread_id & (warpSize - 1);
    int warp_id = thread_id / warpSize;

    int nimgs = envs.nimgs;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int bvk_nbas = envs.nbas * envs.bvk_ncells;
    int ish = bas_ij / bvk_nbas;
    int jsh = bas_ij % bvk_nbas;
    uint32_t img0 = sp_img_offsets[bas_ij];
    int nimgs_j = sp_img_offsets[bas_ij+1] - img0;
    int *ovlp_img_idx = img_idx + img0;
    float ai = diffuse_exps[ish];
    float aj = diffuse_exps[jsh];
    float ci = diffuse_coefs[ish];
    float cj = diffuse_coefs[jsh];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float aij = ai + aj;
    float ai_aij = ai / aij;
    float aj_aij = aj / aij;
    float u = .5f / aij;
    float theta_ij = ai * aj / aij;
    float omega = env[PTR_RANGE_OMEGA];
    if (omega == 0) {
        omega = 0.1f;
    }
    float omega2 = omega * omega;
    float omega_aij = omega2 / (omega2 + aij);
    // fac_guess = log(sqrt(2.x/(omega*sqrt(pi))) * ((2*li+1)*(2*lj+1)*(2*lk+1))**.5/(4*pi)**1.5)
    //           ~ between [0, 2]
    float fac_guess = .5f - logf(omega2)/4;
    // log(ci*cj * (pi/aij)**1.5)
    float log_fac = logf(fabsf(ci*cj)) + 1.717f - 1.5f*logf(aij) + fac_guess;
    // An addiitonal factor for Coulomb integrals
    // log_fac += .25 * logf(2./pi * aij)
    log_fac += .25f * logf(0.6366f * aij);
    log_cutoff -= log_fac;
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;

    int nksh = kidx1 - kidx0;
    extern __shared__ float ak[];
    float *xk = ak + nksh;
    float *yk = ak + nksh * 2;
    float *zk = ak + nksh * 3;
    if (thread_id < nksh) {
        int ksh = ksh_idx[kidx0+thread_id];
        int atom_id = bas[ksh*BAS_SLOTS+ATOM_OF];
        ak[thread_id] = atom_aux_exps[atom_id % envs.natm];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        xk[thread_id] = rk[0];
        yk[thread_id] = rk[1];
        zk[thread_id] = rk[2];
    }
    __syncthreads();

    int nimgs2 = nimgs_j * nimgs;
    for (int img = thread_id; img < nimgs2+thread_id; img += THREADS) {
        bool keep = 0;
        int kL = 0;
        int jL = 0;
        if (img < nimgs2) {
            jL = ovlp_img_idx[img / nimgs];
            kL = img % nimgs;
            float kLx = img_coords[kL*3+0];
            float kLy = img_coords[kL*3+1];
            float kLz = img_coords[kL*3+2];
            float xjLxi = xjxi + img_coords[jL*3+0];
            float yjLyi = yjyi + img_coords[jL*3+1];
            float zjLzi = zjzi + img_coords[jL*3+2];
            float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
            float theta_ij_rr = theta_ij * rr_ij;
            float xij = xjLxi * aj_aij + xi - kLx;
            float yij = yjLyi * aj_aij + yi - kLy;
            float zij = zjLzi * aj_aij + zi - kLz;
            for (int k = 0; k < nksh; ++k) {
                float aij_ak = aij * ak[k];
                float theta = aij_ak * omega2 / (aij_ak + (aij + ak[k]) * omega2);
                float xijk = xij - xk[k];
                float yijk = yij - yk[k];
                float zijk = zij - zk[k];
                float rr = xijk * xijk + yijk * yijk + zijk * zijk;
                float theta_rr = theta * rr + theta_ij_rr;
                if (theta_rr > REMOTE_THRESHOLD) {
                    continue;
                }
                float rt_aij = omega_aij * sqrtf(rr);
                float dr = sqrtf(rr_ij);
                float dri = aj_aij * dr + rt_aij;
                float drj = ai_aij * dr + rt_aij;
                float dri_fac = .5f*li * logf(dri*dri + li*u + 1e-9f);
                float drj_fac = .5f*lj * logf(drj*drj + lj*u + 1e-9f);
                // TODO: an approx dri_fac and drj_fac
                float estimator = dri_fac + drj_fac - theta_rr;
                if (estimator > log_cutoff) {
                    keep = 1;
                    break;
                }
            }
        }
        __syncthreads();
        __shared__ int warp_offsets[WARPS];
        unsigned m = __ballot_sync(0xffffffff, keep);
        int warp_count = __popc(m);
        int lane_index = __popc(m & ((1u << lane) - 1));
        if (lane == 0) {
            warp_offsets[warp_id] = warp_count;
        }
        int pool_offset = img_counts;
        __syncthreads();

        if (thread_id == 0) {
            int sum = 0;
            for (int i = 0; i < WARPS; i++) {
                int temp = warp_offsets[i];
                warp_offsets[i] = sum;
                sum += temp;
            }
            img_counts += sum;
        }
        __syncthreads();
        if (keep) {
            int index = warp_offsets[warp_id] + lane_index;
            img_pool[pool_offset+index] = jL * nimgs + kL;
        }
    }
}
