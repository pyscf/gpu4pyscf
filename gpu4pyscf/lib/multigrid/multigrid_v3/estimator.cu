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

__global__ static
void aft_Ecut_kernel(float *Ecut, float *exps, float *log_cs, int *ls,
                     float *bas_coords, float *lattice_vectors, int *nimgs,
                     int nbas, float log_cutoff, float Ecut_max, int is_mgga)
{
    int bas_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int npairs = nbas * nbas;
    if (bas_ij >= npairs) {
        return;
    }
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int nimgs_x = nimgs[0];
    int nimgs_y = nimgs[1];
    int nimgs_z = nimgs[2];
    int tot_imgs = (nimgs_x*2+1) * (nimgs_y*2+1) * (nimgs_z*2+1);
    size_t supmol_nbas = tot_imgs * nbas;
    size_t ij_off = ish * supmol_nbas + jsh;
    for (int n = 0; n < tot_imgs; ++n) {
        Ecut[n * nbas + ij_off] = 0.f;
    }
    if (ish < jsh) {
        return;
    }

    float ai = exps[ish];
    float aj = exps[jsh];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta = ai * fj;
    int li = ls[ish];
    int lj = ls[jsh];
    if (is_mgga) {
        li += 1;
        lj += 1;
    }
    float log_cicj = log_cs[ish] + log_cs[jsh];
    float log_fac = log_cicj + 1.717f - 1.5f * logf(aij) - log_cutoff;
    log_fac = max(log_fac, 1e-9f);
    float rr_raw = log_fac / theta;
    float Ecut_raw = log_fac * (2*aij);
    float Ecut_2a = Ecut_raw / (2*aij*aij);
    float dri_fac = .5f * logf(.5f*li/aij + fi*fi*rr_raw + Ecut_2a);
    float drj_fac = .5f * logf(.5f*lj/aij + fj*fj*rr_raw + Ecut_2a);
    // An approximate penalty for the polynomial part of the gaussian product
    log_fac += li * dri_fac + lj * drj_fac;
    log_fac = max(log_fac, 0.f);
    float rr_cutoff = log_fac / theta;

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
    float a00 = lattice_vectors[0];
    float a01 = lattice_vectors[1];
    float a02 = lattice_vectors[2];
    float a10 = lattice_vectors[3];
    float a11 = lattice_vectors[4];
    float a12 = lattice_vectors[5];
    float a20 = lattice_vectors[6];
    float a21 = lattice_vectors[7];
    float a22 = lattice_vectors[8];
    for (int ix = -nimgs_x, n = 0; ix <= nimgs_x; ++ix) {
    for (int iy = -nimgs_y; iy <= nimgs_y; ++iy) {
    for (int iz = -nimgs_z; iz <= nimgs_z; ++iz, ++n) {
        float xjLxi = xjxi + ix * a00 + iy * a10 + iz * a20;
        float yjLyi = yjyi + ix * a01 + iy * a11 + iz * a21;
        float zjLzi = zjzi + ix * a02 + iy * a12 + iz * a22;
        float rr = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
        if (rr < rr_cutoff) {
// Ecut estimation based on pyscf.pbc.gto.cell.estimate_ke_cutoff
// Factors for Ecut estimation should be
//     fac = cs[:,None]*cs * cp.exp(-theta*dr**2) * fac_dri * fac_drj * fl
// where
//     fac_dri = (li * .5/aij + dri**2 + Ecut/2/aij**2)**(li*.5)
//             ~= (li * .5/aij + dri**2 + log(1./precision)/aij)**(li*.5)
//     fac_drj = (lj * .5/aij + drj**2 + Ecut/2/aij**2)**(lj*.5)
//             ~= (lj * .5/aij + drj**2 + log(1./precision)/aij)**(lj*.5)
// Here, this fac is approximately derived from the overlap integral
// Ecut ~= log(fac / precision) * 2*aij
            float Ecut_estimate = (log_fac - theta*rr) * (2*aij);
            Ecut[n * nbas + ij_off] = min(Ecut_estimate, Ecut_max);
        }
    } } }
}

extern "C" {
int estimate_aft_Ecut(float *Ecut, float *exps, float *log_cs, int *ls,
                      float *bas_coords, float *lattice_vectors, int *nimgs,
                      int nbas, float log_cutoff, float Ecut_max, int is_mgga)
{
    int threads = 256;
    int blocks = (nbas*nbas + threads-1)/threads;
    aft_Ecut_kernel<<<blocks, threads>>>(
        Ecut, exps, log_cs, ls, bas_coords, lattice_vectors, nimgs, nbas,
        log_cutoff, Ecut_max, is_mgga);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in raw_ovlp_mask: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
