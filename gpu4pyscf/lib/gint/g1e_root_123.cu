/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include "cint2e.cuh"

__global__
static void GINTfill_int3c1e_kernel00(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                      const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                      const double omega, const double* grid_points)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];

    double gout0 = 0;
    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double Px  = x12[ij];
        const double Py  = y12[ij];
        const double Pz  = z12[ij];
        const double PCx = Px - Cx;
        const double PCy = Py - Cy;
        const double PCz = Pz - Cz;
        double a0 = aij;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + aij) : 1.0;
        const double sqrt_theta = omega > 0.0 ? omega / sqrt(omega * omega + aij) : 1.0;
        a0 *= theta;

        const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta;
        const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
        double eri = prefactor;
        if (boys_input > 3.e-7) {
            const double sqrt_boys_input = sqrt(boys_input);
            const double boys_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
            eri *= boys_0;
        }
        gout0 += eri;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    output[i0 + j0 * stride_j + task_grid * stride_ij] = gout0;
}

__global__
static void GINTfill_int3c1e_density_contracted_kernel00(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                         const BasisProdOffsets offsets, const int nprim_ij,
                                                         const double omega, const double* grid_points)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    // const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    // const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    // const int ish = bas_pair2bra[bas_ij];
    // const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];

    double gout0 = 0;
    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double Px  = x12[ij];
        const double Py  = y12[ij];
        const double Pz  = z12[ij];
        const double PCx = Px - Cx;
        const double PCy = Py - Cy;
        const double PCz = Pz - Cz;
        double a0 = aij;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + aij) : 1.0;
        const double sqrt_theta = omega > 0.0 ? omega / sqrt(omega * omega + aij) : 1.0;
        a0 *= theta;

        const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta;
        const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
        double eri = prefactor;
        if (boys_input > 3.e-7) {
            const double sqrt_boys_input = sqrt(boys_input);
            const double boys_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
            eri *= boys_0;
        }
        gout0 += eri;
    }

    const double D = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair];
    atomicAdd(output + task_grid, D * gout0);
}

__global__
static void GINTfill_int3c1e_kernel10(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                      const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                      const double omega, const double* grid_points)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];

    double gout_x = 0;
    double gout_y = 0;
    double gout_z = 0;
    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double Px  = x12[ij];
        const double Py  = y12[ij];
        const double Pz  = z12[ij];
        const double PCx = Px - Cx;
        const double PCy = Py - Cy;
        const double PCz = Pz - Cz;
        const double PAx = Px - Ax;
        const double PAy = Py - Ay;
        const double PAz = Pz - Az;
        double a0 = aij;
        const double one_over_two_p = 0.5 / aij;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + aij) : 1.0;
        const double sqrt_theta = omega > 0.0 ? omega / sqrt(omega * omega + aij) : 1.0;
        a0 *= theta;

        const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta;
        const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
        double eri_x = prefactor;
        double eri_y = prefactor;
        double eri_z = prefactor;
        if (boys_input > 3.e-7) {
            const double sqrt_boys_input = sqrt(boys_input);
            const double R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
            const double R000_1 = -a0 * (R000_0 - exp(-boys_input)) / boys_input;
            eri_x *= R000_0 * PAx + R000_1 * PCx * one_over_two_p;
            eri_y *= R000_0 * PAy + R000_1 * PCy * one_over_two_p;
            eri_z *= R000_0 * PAz + R000_1 * PCz * one_over_two_p;
        }
        gout_x += eri_x;
        gout_y += eri_y;
        gout_z += eri_z;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    output[i0 + 0 + j0 * stride_j + task_grid * stride_ij] = gout_x;
    output[i0 + 1 + j0 * stride_j + task_grid * stride_ij] = gout_y;
    output[i0 + 2 + j0 * stride_j + task_grid * stride_ij] = gout_z;
}

__global__
static void GINTfill_int3c1e_density_contracted_kernel10(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                         const BasisProdOffsets offsets, const int nprim_ij,
                                                         const double omega, const double* grid_points)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    // const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    // const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];

    double gout_x = 0;
    double gout_y = 0;
    double gout_z = 0;
    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double Px  = x12[ij];
        const double Py  = y12[ij];
        const double Pz  = z12[ij];
        const double PCx = Px - Cx;
        const double PCy = Py - Cy;
        const double PCz = Pz - Cz;
        const double PAx = Px - Ax;
        const double PAy = Py - Ay;
        const double PAz = Pz - Az;
        double a0 = aij;
        const double one_over_two_p = 0.5 / aij;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + aij) : 1.0;
        const double sqrt_theta = omega > 0.0 ? omega / sqrt(omega * omega + aij) : 1.0;
        a0 *= theta;

        const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta;
        const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
        double eri_x = prefactor;
        double eri_y = prefactor;
        double eri_z = prefactor;
        if (boys_input > 3.e-7) {
            const double sqrt_boys_input = sqrt(boys_input);
            const double R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
            const double R000_1 = -a0 * (R000_0 - exp(-boys_input)) / boys_input;
            eri_x *= R000_0 * PAx + R000_1 * PCx * one_over_two_p;
            eri_y *= R000_0 * PAy + R000_1 * PCy * one_over_two_p;
            eri_z *= R000_0 * PAz + R000_1 * PCz * one_over_two_p;
        }
        gout_x += eri_x;
        gout_y += eri_y;
        gout_z += eri_z;
    }

    // Density element 0 is the 000 element and is not used in McMurchie-Davidson algorithm. Density element 1~3 is the unchanged z,y,x components.
    const double D_x = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + hermite_density_offsets.n_pair_of_angular_pair * 3];
    const double D_y = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + hermite_density_offsets.n_pair_of_angular_pair * 2];
    const double D_z = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + hermite_density_offsets.n_pair_of_angular_pair * 1];
    atomicAdd(output + task_grid, D_x * gout_x + D_y * gout_y + D_z * gout_z);
}
