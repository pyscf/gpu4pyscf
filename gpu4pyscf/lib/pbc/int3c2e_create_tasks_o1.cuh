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
#include "gvhf-rys/vhf.cuh"

#define THREADS         256
#define WARP_SIZE       32
#define WARPS           8
#define REMOTE_THRESHOLD 50

__device__ __forceinline__
void _filter_ij_images(int& img_counts, int *img_pool, PBCIntEnvVars &envs,
                       uint32_t bas_ij, int ksh0, int ksh1, int li, int lj,
                       uint32_t *bas_ij_idx, float *diffuse_exps, float *diffuse_coefs,
                       float *atom_coords, float *atom_aux_exps, float log_cutoff)
{
    int thread_id = threadIdx.x;
    int lane = thread_id & (warpSize - 1);
    int warp_id = thread_id / warpSize;

    int nimgs = envs.nimgs;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int bvk_nbas = envs.nbas * envs.bvk_ncells;
    int bvk_natm = envs.natm * envs.bvk_ncells;
    int ish = bas_ij / bvk_nbas;
    int jsh = bas_ij % bvk_nbas;
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
    int ksh0_atm = bas[ksh0*BAS_SLOTS+ATOM_OF] - bvk_natm;
    int ksh1_atm = bas[(ksh1-1)*BAS_SLOTS+ATOM_OF] - bvk_natm;
    int nimgs2 = nimgs * nimgs;
    for (int img = thread_id; img < nimgs2+thread_id; img += THREADS) {
        bool keep = 0;
        int iL = 0;
        int jiL = 0;
        if (img < nimgs2) {
            jiL = img / nimgs;
            iL = img - nimgs * jiL;
            float xiL = xi + img_coords[iL*3+0];
            float yiL = yi + img_coords[iL*3+1];
            float ziL = zi + img_coords[iL*3+2];
            float xjxiL = xjxi + img_coords[jiL*3+0];
            float yjyiL = yjyi + img_coords[jiL*3+1];
            float zjziL = zjzi + img_coords[jiL*3+2];
            float xij = xjxiL * aj_aij + xiL;
            float yij = yjyiL * aj_aij + yiL;
            float zij = zjziL * aj_aij + ziL;
            float rr_ij = xjxiL * xjxiL + yjyiL * yjyiL + zjziL * zjziL;
            float theta_ij_rr = theta_ij * rr_ij;
            for (int k = ksh0_atm; k <= ksh1_atm; k++) {
                float ak = atom_aux_exps[k];
                float aij_ak = aij * ak;
                float theta = aij_ak * omega2 / (aij_ak + (aij + ak) * omega2);
                float xk = atom_coords[k*3+0];
                float yk = atom_coords[k*3+1];
                float zk = atom_coords[k*3+2];
                float xijk = xij - xk;
                float yijk = yij - yk;
                float zijk = zij - zk;
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
            img_pool[index] = img;
        }
    }
}
