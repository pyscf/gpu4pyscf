/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
#include "multigrid.cuh"

#define REMOTE_THRESHOLD 50

// An estimation of the upper bound of the overlap |<cell0|supcmol>| for
// shell pairs between the primitve cell and the super-mol
__global__ static
void ovlp_mask_estimation_kernel(int8_t *ovlp_mask, float *Ecut, float *radius,
                                 float *exps, float *log_coeff,
                                 float *bas_coords, int *ao_loc_in_cell0,
                                 int *ls, int cell0_nbas, int nbas,
                                 int hermi, int l_inc, float log_cutoff)
{
    int bas_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int npairs = cell0_nbas * nbas;
    if (bas_ij >= npairs) {
        return;
    }
    ovlp_mask[bas_ij] = 0;
    Ecut[bas_ij] = 0;
    radius[bas_ij] = 0;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    // assume the hermitian symmetry in Coulomb matrix.
    // Note: hermitian symmetry might not be available in methods like TDDFT
    if (hermi && ao_loc_in_cell0[ish] < ao_loc_in_cell0[jsh]) {
        return;
    }

    int li = ls[ish];
    int lj = ls[jsh];
    float ai = exps[ish];
    float aj = exps[jsh];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta = ai * fj;
    float *bas_x = bas_coords;
    float *bas_y = bas_coords + nbas;
    float *bas_z = bas_coords + nbas * 2;
    float xi = bas_x[ish];
    float yi = bas_y[ish];
    float zi = bas_z[ish];
    float xj = bas_x[jsh];
    float yj = bas_y[jsh];
    float zj = bas_z[jsh];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
    if (theta*rr_ij > REMOTE_THRESHOLD) {
        return;
    }
    float dr = sqrtf(rr_ij);
    float dri = fj * dr;
    float drj = fi * dr;
    float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
    float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
    float fac_norm = log_coeff[ish] + log_coeff[jsh] + 1.717f - 1.5f * logf(aij);
    float log_ovlp = fac_norm - theta*rr_ij + dri_fac + drj_fac;
    float log_fac = log_ovlp - log_cutoff;

    if (log_fac > 0) {
        ovlp_mask[bas_ij] = 1;
        // Ecut estimation based on pyscf.pbc.gto.cell.estimate_ke_cutoff
        // Factors for Ecut estimation should be
        //     fac = cs[:,None]*cs * cp.exp(-theta*dr**2) * fac_dri * fac_drj * fl
        // where
        //     fac_dri = (li * .5/aij + dri**2 + Ecut/2/aij**2)**(li*.5)
        //             ~= (li * .5/aij + dri**2 + log(1./precision)/aij)**(li*.5)
        //     fac_drj = (lj * .5/aij + drj**2 + Ecut/2/aij**2)**(lj*.5)
        //             ~= (lj * .5/aij + drj**2 + log(1./precision)/aij)**(lj*.5)
        // Here, this fac is approximately derived from the overlap integral
        // fac = ovlp / precision
        // Ecut = cp.log(fac + 1.) * 2*aij
        Ecut[bas_ij] = log_fac * (2*aij);
        // Estimate radius:
        // rho[r-Rp] = fl*cs[:cell0_nprims,None]*cs * exp(-theta*dr**2)
        //             * r**lij * exp(-aij*r**2)
        // radius = (cp.log(ovlp/precision * radius**(lij+l_inc) + 1.) / aij)**.5
        // radius = (cp.log(ovlp/precision * radius**(lij+l_inc) + 1.) / aij)**.5
        float r = 2.;
        int lij = li + lj;
        r = (log_fac + (lij+l_inc)*logf(r)) / aij;
        if (r < 0) {
            return;
        }
        r = sqrtf(r);
        r = (log_fac + (lij+l_inc)*logf(r)) / aij;
        if (r < 0) {
            return;
        }
        radius[bas_ij] = sqrtf(r);
    }
}

__global__ static
void filter_supmol_bas_kernel(int8_t *mask, double *Ls, int nimgs,
                              int *uniq_Dbasis_idx, int nbas_uniq,
                              int *bas, int nbas, double *env, float log_cutoff)
{
    int jsh = blockIdx.x * blockDim.x + threadIdx.x + nbas;
    if (jsh >= nbas*nimgs) {
        return;
    }

    int img = jsh / nbas;
    int cell0_jsh = jsh % nbas;
    int lj = bas[PRIMBAS_ANG+cell0_jsh*PRIMBAS_SLOTS];
    float aj = env[bas[PRIMBAS_EXP+cell0_jsh*PRIMBAS_SLOTS]];
    float cj = env[bas[PRIMBAS_COEFF+cell0_jsh*PRIMBAS_SLOTS]];
    double *rj = env + bas[PRIMBAS_COORD+cell0_jsh*PRIMBAS_SLOTS];
    float xj = rj[0] + Ls[img*3+0];
    float yj = rj[1] + Ls[img*3+1];
    float zj = rj[2] + Ls[img*3+2];
    for (int n = 0; n < nbas_uniq; ++n) {
        int ish = uniq_Dbasis_idx[n];
        int li = bas[PRIMBAS_ANG+ish*PRIMBAS_SLOTS];;
        float ai = env[bas[PRIMBAS_EXP+ish*PRIMBAS_SLOTS]];;
        float aij = ai + aj;
        float fi = ai / aij;
        float fj = aj / aij;
        float theta = ai * fj;
        double *ri = env + bas[PRIMBAS_COORD+ish*PRIMBAS_SLOTS];
        float xi = ri[0];
        float yi = ri[1];
        float zi = ri[2];
        float xjxi = xj - xi;
        float yjyi = yj - yi;
        float zjzi = zj - zi;
        float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
        float theta_rr = theta * rr_ij;
        if (theta*rr_ij > REMOTE_THRESHOLD) {
            continue;
        }
        float ci = env[bas[PRIMBAS_COEFF+ish*PRIMBAS_SLOTS]];
        float dr = sqrtf(rr_ij);
        float dri = fj * dr;
        float drj = fi * dr;
        float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
        float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
        float fac_norm = logf(fabsf(ci * cj)) + 1.717f - 1.5f * logf(aij);
        float s = fac_norm - theta_rr + dri_fac + drj_fac;
        if (s > log_cutoff) {
            mask[jsh] = 1;
            return;
        }
    }
    mask[jsh] = 0;
}

extern "C" {
int ovlp_mask_estimation(int8_t *ovlp_mask, float *Ecut, float *radius,
                         float *exps, float *log_coeff,
                         float *bas_coords, int *ao_loc_in_cell0,
                         int *ls, int cell0_nbas, int nbas,
                         int hermi, int l_inc, float log_cutoff)
{
    constexpr int threads = 1024;
    int blocks = (cell0_nbas*nbas + threads-1)/threads;
    ovlp_mask_estimation_kernel<<<blocks, threads>>>(
        ovlp_mask, Ecut, radius, exps, log_coeff, bas_coords, ao_loc_in_cell0,
        ls, cell0_nbas, nbas, hermi, l_inc, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in overlap_estimation: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int filter_supmol_bas(int8_t *mask, double *Ls, int nimgs,
                      int *uniq_Dbasis_idx, int nbas_uniq,
                      int *bas, int nbas, double *env, float log_cutoff)
{
    constexpr int threads = 1024;
    int blocks = (nbas*nimgs + threads-1)/threads;
    filter_supmol_bas_kernel<<<blocks, threads>>>(
        mask, Ls, nimgs, uniq_Dbasis_idx, nbas_uniq, bas, nbas, env, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in filter_supmol_bas: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
