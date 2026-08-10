/*
 * Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "constant_objects.cuh"
#include "cartesian.cuh"
#include "utils.cuh"

#define TILE    4
#define THREADS 256

template <typename T>
__device__ static
T estimate_rcut(int li, int lj, T x, T aij, T xpi, T xpj, T log_factor)
{
    // let s = r - Rp
    // rho[r-Rp] ~ ci*cj * exp(-theta*(Ri-Rj)**2) * (s+Rp-Ri)**li * (s+Rp-Rj)**lj * exp(-aij*s**2)
    //           ~= ovlp * (s+Rp-Ri)**li * (s+Rp-Rj)**lj * exp(-aij*s**2)
    // radius can be solved using fixed iteration
    // radius = (log(ovlp/precision * (s+Rpi)**li * (s+Rpj)**lj) / aij)**.5
    T aij_ss = log_factor + li * std::log(x + std::abs(xpi)) + lj * std::log(x + std::abs(xpj));
    return std::sqrt(max(aij_ss, static_cast<T>(0)) / aij);
}

template <typename T>
__device__ inline
void accumulate(T lower, T upper, T c, T& min_val, T& max_val)
{
    T a = c * lower;
    T b = c * upper;
    min_val += min(a, b);
    max_val += max(a, b);
}

__global__ static
void grid_ranges_kernel(float2 *grid_frac_ranges, float *pair_ke,
                        float *Ecut_by_shell, PBCIntEnvVars envs,
                        int64_t *bas_ij_idx, int li_inc, int lj_inc,
                        int npairs, float log_threshold)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) return;

    int *bas = envs.bas;
    double *env = envs.env;
    int nbas = envs.nbas;
    int bvk_nbas = envs.nbas * envs.bvk_ncells;
    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / NBAS_MAX;
    int jsh = bas_ij % NBAS_MAX;
    int jL = jsh / bvk_nbas;
    jsh = jsh % bvk_nbas;
    // li_inc and lj_inc to account for derivatives
    int li = bas[ish*BAS_SLOTS+ANG_OF] + li_inc;
    int lj = bas[jsh*BAS_SLOTS+ANG_OF] + lj_inc;
    int expi = bas[ish*BAS_SLOTS+PTR_EXP];
    int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float ai = env[expi];
    float aj = env[expj];
    float aij = ai + aj;
    float aj_aij = aj / aij;
    float xi = env[ri+0];
    float yi = env[ri+1];
    float zi = env[ri+2];
    float xj = env[rj+0] + envs.img_coords[jL*3+0];
    float yj = env[rj+1] + envs.img_coords[jL*3+1];
    float zj = env[rj+2] + envs.img_coords[jL*3+2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float rr = distance_squared(xjxi, yjyi, zjzi);
    float xp = xjxi * aj_aij + xi;
    float yp = yjyi * aj_aij + yi;
    float zp = zjzi * aj_aij + zi;
    float theta_rr = ai * aj_aij * rr;
    float ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    float cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    float log_cicj = logf(fabsf(ci * cj));
    float derivative_penalty = li_inc * logf(2*ai) + lj_inc * logf(2*aj);
    float xpi = xp - xi;
    float xpj = xp - xj;
    float ypi = yp - yi;
    float ypj = yp - yj;
    float zpi = zp - zi;
    float zpj = zp - zj;

    // let s = r - Rp
    // rho[r-Rp] ~ ci*cj * exp(-theta*(Ri-Rj)**2) * (s+Rp-Ri)**li * (s+Rp-Rj)**lj * exp(-aij*s**2)
    float log_factor = log_cicj + 1.717f - 1.5f * logf(aij) - log_threshold - theta_rr;
    log_factor += derivative_penalty;
    // initial guess
    float log_r = 2.303f; // log(10)
    float radius = sqrtf(max(log_factor + (li+lj)*log_r, 1e-20f) / aij);
    // to encounter the integral over remaining space ~ int_radius^inf 4*pi*r^2 exp(-aij*r^2);
    float penalty = 6.283f * radius / (2*aij);
    // To accurately integrate a gaussian, the required resolution (Ngrid/Bohr) ~ 2*a**.5
    float resolution = 2*sqrtf(aij);
    log_factor += max(logf(max(penalty, 12.56f*radius*radius/resolution)), 0.f);
    float x_cut = estimate_rcut(li, lj, radius, aij, xpi, xpj, log_factor);
    float y_cut = estimate_rcut(li, lj, radius, aij, ypi, ypj, log_factor);
    float z_cut = estimate_rcut(li, lj, radius, aij, zpi, zpj, log_factor);

    float b00 = c_reciprocal_lattice_vectors[0];
    float b01 = c_reciprocal_lattice_vectors[1];
    float b02 = c_reciprocal_lattice_vectors[2];
    float b10 = c_reciprocal_lattice_vectors[3];
    float b11 = c_reciprocal_lattice_vectors[4];
    float b12 = c_reciprocal_lattice_vectors[5];
    float b20 = c_reciprocal_lattice_vectors[6];
    float b21 = c_reciprocal_lattice_vectors[7];
    float b22 = c_reciprocal_lattice_vectors[8];

    float xp_frac = xp * b00 + yp * b01 + zp * b02;
    float yp_frac = xp * b10 + yp * b11 + zp * b12;
    float zp_frac = xp * b20 + yp * b21 + zp * b22;

    float xcut_frac = x_cut * fabsf(b00) + y_cut * fabsf(b01) + z_cut * fabsf(b02);
    float ycut_frac = x_cut * fabsf(b10) + y_cut * fabsf(b11) + z_cut * fabsf(b12);
    float zcut_frac = x_cut * fabsf(b20) + y_cut * fabsf(b21) + z_cut * fabsf(b22);

    float2 *xfrac_range = grid_frac_ranges;
    float2 *yfrac_range = grid_frac_ranges + npairs;
    float2 *zfrac_range = grid_frac_ranges + npairs * 2;
    xfrac_range[pair_id] = {xp_frac - xcut_frac, xp_frac + xcut_frac};
    yfrac_range[pair_id] = {yp_frac - ycut_frac, yp_frac + ycut_frac};
    zfrac_range[pair_id] = {zp_frac - zcut_frac, zp_frac + zcut_frac};

    // When cutoff radius is 0, the contribution of this orbital pair is small.
    // By setting its pair_ke to 0, this orbital pair will be discarded when
    // filtering orbitals in _partition_ke_for_fft function.
    if (x_cut < 1e-3 || y_cut < 1e-3 || z_cut < 1e-3) {
        pair_ke[pair_id] = -1.f;
    } else {
        float ish_ke = Ecut_by_shell[ish];
        float jsh_ke = Ecut_by_shell[jsh % nbas];
        pair_ke[pair_id] = max(ish_ke, jsh_ke);
    }
}

__global__ static
void grid_range_to_tiles_kernel(int *grid_tile_idx, int64_t *dressed_bas_ij,
                                int64_t *bas_ij_idx, float2 *grid_frac_ranges,
                                int nimgs_x, int nimgs_y, int nimgs_z,
                                int mesh_x, int mesh_y, int mesh_z, int npairs,
                                int nbas, int *head)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) return;

    int64_t bas_ij = bas_ij_idx[pair_id];
    int64_t bas_ij_stride = nbas * NBAS_MAX;

    float2 range = grid_frac_ranges[pair_id];
    float xfrac_lower = range.x;
    float xfrac_upper = range.y;
    range = grid_frac_ranges[npairs+pair_id];
    float yfrac_lower = range.x;
    float yfrac_upper = range.y;
    range = grid_frac_ranges[npairs*2+pair_id];
    float zfrac_lower = range.x;
    float zfrac_upper = range.y;

    int tiles_x = (mesh_x + TILE - 1) / TILE;
    int tiles_y = (mesh_y + TILE - 1) / TILE;
    int tiles_z = (mesh_z + TILE - 1) / TILE;
    int tile_size_x = min(TILE, mesh_x);
    int tile_size_y = min(TILE, mesh_y);
    int tile_size_z = min(TILE, mesh_z);

    int img_x_lower = floor(xfrac_lower);
    int img_y_lower = floor(yfrac_lower);
    int img_z_lower = floor(zfrac_lower);
    int img_x_upper = floor(xfrac_upper);
    int img_y_upper = floor(yfrac_upper);
    int img_z_upper = floor(zfrac_upper);
    img_x_lower = max(img_x_lower, -nimgs_x);
    img_y_lower = max(img_y_lower, -nimgs_y);
    img_z_lower = max(img_z_lower, -nimgs_z);
    img_x_upper = min(img_x_upper,  nimgs_x);
    img_y_upper = min(img_y_upper,  nimgs_y);
    img_z_upper = min(img_z_upper,  nimgs_z);
    int rem_x_lower = floor(max(xfrac_lower - img_x_lower, 0.f) * mesh_x / tile_size_x);
    int rem_y_lower = floor(max(yfrac_lower - img_y_lower, 0.f) * mesh_y / tile_size_y);
    int rem_z_lower = floor(max(zfrac_lower - img_z_lower, 0.f) * mesh_z / tile_size_z);
    int rem_x_upper = ceil (min(xfrac_upper - img_x_upper, 1.f) * mesh_x / tile_size_x);
    int rem_y_upper = ceil (min(yfrac_upper - img_y_upper, 1.f) * mesh_y / tile_size_y);
    int rem_z_upper = ceil (min(zfrac_upper - img_z_upper, 1.f) * mesh_z / tile_size_z);
    int count_x = rem_x_upper - rem_x_lower + (img_x_upper - img_x_lower) * tiles_x;
    int count_y = rem_y_upper - rem_y_lower + (img_y_upper - img_y_lower) * tiles_y;
    int count_z = rem_z_upper - rem_z_lower + (img_z_upper - img_z_lower) * tiles_z;
    // TODO: tiles in the corners sometimes are out of the cutoff radius.
    // They can be discarded and counts can be reduced
    int counts = count_x * count_y * count_z;
    int n = atomicAdd(head, counts);
    int Ny = nimgs_y * 2 + 1;
    int Nz = nimgs_z * 2 + 1;
    // lattice sum spans over [-nimgs_x, nimgs_x], [-nimgs_y, nimgs_y], [-nimgs_z, nimgs_z], 
    // Add img_offset to avoid negative indexing
    int img_offset = nimgs_x * Ny * Nz + nimgs_y * Nz + nimgs_z;
    for (int x = rem_x_lower, img_x = img_x_lower; x < rem_x_upper || img_x < img_x_upper;) {
        for (int y = rem_y_lower, img_y = img_y_lower; y < rem_y_upper || img_y < img_y_upper;) {
            for (int z = rem_z_lower, img_z = img_z_lower; z < rem_z_upper || img_z < img_z_upper;) {
                // when (x, y, z) lies out of the unit cell, they can be repositioned
                // by shifting the lattice sum index on bra
                int64_t latsum_idx = img_offset + (img_x * Ny + img_y) * Nz + img_z;
                // dressed_bas_ij stores (latsum_idx*nbas+ish, jL*bvk_nbas+jsh).
                // latsum_idx is the image index to reposition bra whereas jL is
                // the image index relative to bra.
                dressed_bas_ij[n] = latsum_idx * bas_ij_stride + bas_ij;
                grid_tile_idx[n] = (x * tiles_y + y) * tiles_z + z;

                n++;
                z++;
                if (z >= tiles_z && img_z < img_z_upper) {
                    z = 0;
                    img_z++;
                }
            }
            y++;
            if (y >= tiles_y && img_y < img_y_upper) {
                y = 0;
                img_y++;
            }
        }
        x++;
        if (x >= tiles_x && img_x < img_x_upper) {
            x = 0;
            img_x++;
        }
    }
}

// An estimation of the upper bound of the overlap |<cell0|supcmol>| for
// shell pairs between the primitve cell and the super-mol
__global__ static
void ovlp_mask_estimation_kernel(int8_t *ovlp_mask, PBCIntEnvVars envs,
                                 double *img_coords, int nimgs, float log_cutoff)
{
    int jsh = blockIdx.x * blockDim.x + threadIdx.x;
    int ish = blockIdx.y * blockDim.y + threadIdx.y;
    int nbas = envs.nbas;
    int bvk_nbas = envs.nbas * envs.bvk_ncells;
    if (ish >= nbas || jsh >= bvk_nbas) {
        return;
    }
    int jsh_cell0 = jsh % nbas;
    if (ish < jsh_cell0) {
        return;
    }
    int *bas = envs.bas;
    double *env = envs.env;
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    float ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
    float aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
    float ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    float cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta = ai * fj;
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;

    float log_cicj = logf(fabsf(ci * cj));
    float log_fac = log_cicj + 1.717f - 1.5f * logf(aij) - log_cutoff;
    log_fac = max(log_fac, 1e-9f);
    float rr_raw = log_fac / theta;
    float dri_fac = .5f * logf(.5f*li/aij + fi*fi*rr_raw);
    float drj_fac = .5f * logf(.5f*lj/aij + fj*fj*rr_raw);
    // An approximate penalty for the polynomial part of the gaussian product
    log_fac += li * dri_fac + lj * drj_fac;
    log_fac = max(log_fac, 0.f);
    float rr_cutoff = log_fac / theta;

    for (int img = 0; img < nimgs; ++img) {
        float xjLxi = xjxi + img_coords[img*3+0];
        float yjLyi = yjyi + img_coords[img*3+1];
        float zjLzi = zjzi + img_coords[img*3+2];
        float rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
        if (rr_ij < rr_cutoff) {
            ovlp_mask[ish * bvk_nbas + jsh] = 1;
            break;
        }
    }
}

__global__ static
void estimate_aft_Ecut_kernel(float *Ecut, int64_t *bas_ij_idx, PBCIntEnvVars envs,
                              double *img_coords, int nimgs, int npairs,
                              float log_cutoff, float Ecut_max, int is_mgga)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) {
        return;
    }
    int *bas = envs.bas;
    double *env = envs.env;
    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / NBAS_MAX;
    int jsh = bas_ij % NBAS_MAX;
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    float ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
    float aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
    float ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    float cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta = ai * fj;
    if (is_mgga) {
        li += 1;
        lj += 1;
        ci *= ai * 2;
        cj *= aj * 2;
    }
    float log_cicj = logf(fabsf(ci * cj));
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

    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    float Ecut_required = 0.f;
    for (int img = 0; img < nimgs; ++img) {
        float xjLxi = xjxi + img_coords[img*3+0];
        float yjLyi = yjyi + img_coords[img*3+1];
        float zjLzi = zjzi + img_coords[img*3+2];
        float rr = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
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
        Ecut_required = max(Ecut_estimate, Ecut_required);
    }
    Ecut[pair_id] = min(Ecut_max, Ecut_required);
}

__global__ static
void supmol_non_trivial_pairs_kernel(int64_t *supmol_bas_ij, int64_t *bas_ij_idx,
                                     PBCIntEnvVars envs, int npairs, float log_cutoff,
                                     int is_mgga, int *head)
{
    int thread_id = threadIdx.x;
    int pair_id = blockIdx.x * blockDim.x + thread_id;
    if (pair_id >= npairs) {
        return;
    }
    constexpr int batch_size = 64;
    __shared__ int8_t img_cache[THREADS*batch_size];
    int bvk_nbas = envs.nbas * envs.bvk_ncells;
    int nimgs = envs.nimgs;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;

    int64_t bas_ij = bas_ij_idx[pair_id];
    int ish = bas_ij / NBAS_MAX;
    int jsh = bas_ij % NBAS_MAX;
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    float ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
    float aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
    float ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    float cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta = ai * fj;
    if (is_mgga) {
        li += 1;
        lj += 1;
        ci *= ai * 2;
        cj *= aj * 2;
    }
    float log_cicj = logf(fabsf(ci * cj));
    float log_fac = log_cicj + 1.717f - 1.5f * logf(aij) - log_cutoff;
    log_fac = max(log_fac, 1e-9f);
    float rr_raw = log_fac / theta;
    float dri_fac = .5f * logf(.5f*li/aij + fi*fi*rr_raw);
    float drj_fac = .5f * logf(.5f*lj/aij + fj*fj*rr_raw);
    // An approximate penalty for the polynomial part of the gaussian product
    log_fac += li * dri_fac + lj * drj_fac;
    log_fac = max(log_fac, 0.f);
    float rr_cutoff = log_fac / theta;

    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    float xjxi = xj - xi;
    float yjyi = yj - yi;
    float zjzi = zj - zi;
    for (int img0 = 0; img0 < nimgs; img0 += batch_size) {
        int count = 0;
        for (int i = 0; i < min(batch_size, nimgs-img0); ++i) {
            int img = img0 + i;
            float xjLxi = xjxi + img_coords[img*3+0];
            float yjLyi = yjyi + img_coords[img*3+1];
            float zjLzi = zjzi + img_coords[img*3+2];
            float rr = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
            if (rr < rr_cutoff) {
                img_cache[count*THREADS+thread_id] = i;
                count++;
            }
        }
        if (count > 0) {
            int off = atomicAdd(head, count);
            for (int n = 0; n < count; ++n) {
                int64_t img = img0 + img_cache[n*THREADS+thread_id];
                // the jsh Id in bas_ij is updated to img*bvk_nbas+jsh
                supmol_bas_ij[off+n] = img * bvk_nbas + bas_ij;
            }
        }
    }
}

extern "C" {
int gaussian_prod_grid_ranges(float2 *grid_frac_ranges, float *pair_ke,
                              float *Ecut_by_shell, PBCIntEnvVars *envs,
                              int64_t *bas_ij_idx, int npairs,
                              int li_inc, int lj_inc, float log_threshold)
{
    int batches = (npairs + THREADS-1) / THREADS;
    grid_ranges_kernel<<<batches, THREADS>>>(
        grid_frac_ranges, pair_ke, Ecut_by_shell, *envs, bas_ij_idx,
        li_inc, lj_inc, npairs, log_threshold);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in gaussian_prod_grid_ranges: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int grid_range_to_tiles(int *grid_tile_idx, int64_t *dressed_bas_ij,
                        int64_t *bas_ij_idx, float2 *grid_frac_ranges,
                        int *nimgs, int *mesh, int npairs, int nbas, int *head)
{
    cudaMemset(head, 0, sizeof(int));
    int nimgs_x = nimgs[0];
    int nimgs_y = nimgs[1];
    int nimgs_z = nimgs[2];
    int mesh_x = mesh[0];
    int mesh_y = mesh[1];
    int mesh_z = mesh[2];
    int batches = (npairs + THREADS-1) / THREADS;
    grid_range_to_tiles_kernel<<<batches, THREADS>>>(
        grid_tile_idx, dressed_bas_ij, bas_ij_idx, grid_frac_ranges,
        nimgs_x, nimgs_y, nimgs_z, mesh_x, mesh_y, mesh_z, npairs, nbas, head);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in grid_range_to_tiles: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int bvk_ovlp_mask_estimation(int8_t *ovlp_mask, PBCIntEnvVars *envs,
                             double *img_coords, int nimgs, float log_cutoff)
{
    int nbas = envs->nbas;
    int bvk_nbas = nbas * envs->bvk_ncells;
    dim3 threads(16, 16);
    dim3 blocks((bvk_nbas + 15) / 16, (nbas + 15) / 16);
    ovlp_mask_estimation_kernel<<<blocks, threads>>>(
            ovlp_mask, *envs, img_coords, nimgs, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bvk_ovlp_mask_estimation: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
int supmol_non_trivial_pairs(int64_t *supmol_bas_ij, int64_t *bas_ij_idx,
                             PBCIntEnvVars *envs, int npairs, float log_cutoff,
                             int is_mgga, int *head)
{
    cudaMemset(head, 0, sizeof(int));
    int blocks = (npairs + THREADS-1)/THREADS;
    supmol_non_trivial_pairs_kernel<<<blocks, THREADS>>>(
            supmol_bas_ij, bas_ij_idx, *envs, npairs, log_cutoff, is_mgga, head);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in bvk_ovlp_mask_estimation: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int estimate_aft_Ecut(float *Ecut, int64_t *bas_ij_idx, PBCIntEnvVars *envs,
                      double *img_coords, int nimgs, int npairs,
                      float log_cutoff, float Ecut_max, int is_mgga)
{
    int blocks = (npairs + THREADS-1)/THREADS;
    estimate_aft_Ecut_kernel<<<blocks, THREADS>>>(
        Ecut, bas_ij_idx, *envs, img_coords, nimgs, npairs, log_cutoff,
        Ecut_max, is_mgga);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in raw_ovlp_mask: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
