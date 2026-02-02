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

#define PAGE_SIZE       30
#define REMOTE_THRESHOLD 50
// approximately, 15000*2 images in each ijk shell triplet for 256 threads
#define PAGES_PER_BLOCK  262144

typedef struct {
    int pair_ij;
    uint16_t k;
    uint16_t nimgs;
    uint16_t img_j[PAGE_SIZE];
    uint16_t img_k[PAGE_SIZE];
} ImgIdxPage;

static inline __device__ __forceinline__
void _filter_images(int& num_pages, ImgIdxPage *page_pool, PBCIntEnvVars &envs,
                    int pair_ij, int ksh, int k_id, int li, int lj,
                    uint32_t *bas_ij_idx, int *img_idx, uint32_t *sp_img_offsets,
                    float *diffuse_exps, float *diffuse_coefs, float log_cutoff)
{
    int nimgs = envs.nimgs;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    uint32_t img0 = sp_img_offsets[pair_ij];
    int nimgs_j = sp_img_offsets[pair_ij+1] - img0;
    int *ovlp_img_idx = img_idx + img0;
    float ai = diffuse_exps[ish];
    float aj = diffuse_exps[jsh];
    float ak = diffuse_exps[ksh];
    float ci = diffuse_coefs[ish];
    float cj = diffuse_coefs[jsh];
    float ck = diffuse_coefs[ksh];
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
    float log_fac = logf(fabsf(ci*cj*ck)) + 1.717f - 1.5f*logf(aij) + fac_guess;
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

    float aij_ak = aij * ak;
    float theta = aij_ak * omega2 / (aij_ak + (aij + ak) * omega2);
    double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
    float xk = rk[0];
    float yk = rk[1];
    float zk = rk[2];
    float xixk = xi - xk;
    float yiyk = yi - yk;
    float zizk = zi - zk;
    ImgIdxPage *page = NULL;
    int counts = PAGE_SIZE;
    for (int img = 0; img < nimgs_j*nimgs; ++img) {
        int jL = ovlp_img_idx[img / nimgs];
        int kL = img % nimgs;
        float xixkL = xixk - img_coords[kL*3+0];
        float yiykL = yiyk - img_coords[kL*3+1];
        float zizkL = zizk - img_coords[kL*3+2];
        float xjLxi = xjxi + img_coords[jL*3+0];
        float yjLyi = yjyi + img_coords[jL*3+1];
        float zjLzi = zjzi + img_coords[jL*3+2];
        float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
        float theta_ij_rr = theta_ij * rr_ij;
        float xijk = xjLxi * aj_aij + xixkL;
        float yijk = yjLyi * aj_aij + yiykL;
        float zijk = zjLzi * aj_aij + zizkL;
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
            if (counts == PAGE_SIZE) {
                if (page != NULL) {
                    page->nimgs = PAGE_SIZE;
                }
                int page_offset = atomicAdd(&num_pages, 1);
                if (page_offset >= PAGES_PER_BLOCK) {
                    printf("Page overflow\n");
                    __trap();
                }
                page = page_pool + page_offset;
                page->pair_ij = pair_ij;
                page->k = k_id;
                counts = 0;
            }
            page->img_j[counts] = jL;
            page->img_k[counts] = kL;
            counts++;
        }
    }
    if (page != NULL) {
        page->nimgs = counts;
    }
}
