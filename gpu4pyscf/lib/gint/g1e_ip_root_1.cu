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

#include <math.h>
#include "cint2e.cuh"

__global__
static void GINTfill_int3c1e_ip_kernel00(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                         const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                         const double omega, const double* grid_points, const double* charge_exponents)
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

    const double* __restrict__ a_exponents = c_bpcache.a1;
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
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double deri_dAx = 0;
    double deri_dAy = 0;
    double deri_dAz = 0;
    double deri_dCx = 0;
    double deri_dCy = 0;
    double deri_dCz = 0;
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
        const double minus_two_a = -2.0 * a_exponents[ij];
        const double one_over_two_p = 0.5 / aij;
        double a0 = aij;
        const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
        const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
        a0 *= q_over_p_plus_q;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
        const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
        a0 *= theta;

        const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
        const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
        double R000_0 = 1;
        double R000_1 = -a0 / 3;
        if (boys_input > 1e-14) {
            const double sqrt_boys_input = sqrt(boys_input);
            R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
            R000_1 = -a0 * (R000_0 - exp(-boys_input)) / boys_input;
        }
        const double R100_0 = R000_1 * PCx;
        const double R010_0 = R000_1 * PCy;
        const double R001_0 = R000_1 * PCz;
        deri_dAx += prefactor * minus_two_a * (PAx * R000_0 + one_over_two_p * R100_0);
        deri_dAy += prefactor * minus_two_a * (PAy * R000_0 + one_over_two_p * R010_0);
        deri_dAz += prefactor * minus_two_a * (PAz * R000_0 + one_over_two_p * R001_0);
        deri_dCx += prefactor * R100_0;
        deri_dCy += prefactor * R010_0;
        deri_dCz += prefactor * R001_0;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    output[i0 + j0 * stride_j + task_grid * stride_ij + 0 * stride_ij * ngrids] = deri_dAx;
    output[i0 + j0 * stride_j + task_grid * stride_ij + 1 * stride_ij * ngrids] = deri_dAy;
    output[i0 + j0 * stride_j + task_grid * stride_ij + 2 * stride_ij * ngrids] = deri_dAz;
    output[i0 + j0 * stride_j + task_grid * stride_ij + 3 * stride_ij * ngrids] = deri_dCx;
    output[i0 + j0 * stride_j + task_grid * stride_ij + 4 * stride_ij * ngrids] = deri_dCy;
    output[i0 + j0 * stride_j + task_grid * stride_ij + 5 * stride_ij * ngrids] = deri_dCz;
}

__global__
static void GINTfill_int3c1e_ip1_charge_contracted_kernel00(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                                            const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                            const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_ij >= ntasks_ij) {
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

    const double* __restrict__ a_exponents = c_bpcache.a1;
    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];

    double deri_dAx_grid_sum = 0;
    double deri_dAy_grid_sum = 0;
    double deri_dAz_grid_sum = 0;
    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double Cx = grid_point[0];
        const double Cy = grid_point[1];
        const double Cz = grid_point[2];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double deri_dAx_per_grid = 0;
        double deri_dAy_per_grid = 0;
        double deri_dAz_per_grid = 0;
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
            const double minus_two_a = -2.0 * a_exponents[ij];
            const double one_over_two_p = 0.5 / aij;
            double a0 = aij;
            const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
            const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
            a0 *= q_over_p_plus_q;
            const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
            const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
            a0 *= theta;

            const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
            const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
            double R000_0 = 1;
            double R000_1 = -a0 / 3;
            if (boys_input > 1e-14) {
                const double sqrt_boys_input = sqrt(boys_input);
                R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
                R000_1 = -a0 * (R000_0 - exp(-boys_input)) / boys_input;
            }
            deri_dAx_per_grid += prefactor * minus_two_a * (PAx * R000_0 + one_over_two_p * R000_1 * PCx);
            deri_dAy_per_grid += prefactor * minus_two_a * (PAy * R000_0 + one_over_two_p * R000_1 * PCy);
            deri_dAz_per_grid += prefactor * minus_two_a * (PAz * R000_0 + one_over_two_p * R000_1 * PCz);
        }

        const double charge = grid_point[3];
        deri_dAx_grid_sum += deri_dAx_per_grid * charge;
        deri_dAy_grid_sum += deri_dAy_per_grid * charge;
        deri_dAz_grid_sum += deri_dAz_per_grid * charge;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    atomicAdd(output + (i0 + j0 * stride_j + 0 * stride_ij), deri_dAx_grid_sum);
    atomicAdd(output + (i0 + j0 * stride_j + 1 * stride_ij), deri_dAy_grid_sum);
    atomicAdd(output + (i0 + j0 * stride_j + 2 * stride_ij), deri_dAz_grid_sum);
}

__global__
static void GINTfill_int3c1e_ip1_density_contracted_kernel00(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                                             const double* density, const int* aoslice, const int nao,
                                                             const double omega, const double* grid_points, const double* charge_exponents)
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

    const double* __restrict__ a_exponents = c_bpcache.a1;
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
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double deri_dAx = 0;
    double deri_dAy = 0;
    double deri_dAz = 0;
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
        const double minus_two_a = -2.0 * a_exponents[ij];
        const double one_over_two_p = 0.5 / aij;
        double a0 = aij;
        const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
        const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
        a0 *= q_over_p_plus_q;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
        const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
        a0 *= theta;

        const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
        const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
        double R000_0 = 1;
        double R000_1 = -a0 / 3;
        if (boys_input > 1e-14) {
            const double sqrt_boys_input = sqrt(boys_input);
            R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
            R000_1 = -a0 * (R000_0 - exp(-boys_input)) / boys_input;
        }
        deri_dAx += prefactor * minus_two_a * (PAx * R000_0 + one_over_two_p * R000_1 * PCx);
        deri_dAy += prefactor * minus_two_a * (PAy * R000_0 + one_over_two_p * R000_1 * PCy);
        deri_dAz += prefactor * minus_two_a * (PAz * R000_0 + one_over_two_p * R000_1 * PCz);
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish];
    const int j0 = ao_loc[jsh];

    const double Dij = density[i0 + j0 * nao];
    deri_dAx *= Dij;
    deri_dAy *= Dij;
    deri_dAz *= Dij;

    const int i_atom = aoslice[ish];
    atomicAdd(output + (task_grid + ngrids * (i_atom * 3 + 0)), deri_dAx);
    atomicAdd(output + (task_grid + ngrids * (i_atom * 3 + 1)), deri_dAy);
    atomicAdd(output + (task_grid + ngrids * (i_atom * 3 + 2)), deri_dAz);
}

__global__
static void GINTfill_int3c1e_ip2_density_contracted_kernel00(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                             const BasisProdOffsets offsets, const int nprim_ij,
                                                             const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_grid >= ngrids) {
        return;
    }

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double deri_dCx_pair_sum = 0.0;
    double deri_dCy_pair_sum = 0.0;
    double deri_dCz_pair_sum = 0.0;
    for (int task_ij = blockIdx.x * blockDim.x + threadIdx.x; task_ij < ntasks_ij; task_ij += gridDim.x * blockDim.x) {
        const int bas_ij = offsets.bas_ij + task_ij;
        const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;

        const double* __restrict__ a12 = c_bpcache.a12;
        const double* __restrict__ e12 = c_bpcache.e12;
        const double* __restrict__ x12 = c_bpcache.x12;
        const double* __restrict__ y12 = c_bpcache.y12;
        const double* __restrict__ z12 = c_bpcache.z12;

        double deri_dCx_per_pair = 0;
        double deri_dCy_per_pair = 0;
        double deri_dCz_per_pair = 0;
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
            const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
            const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
            a0 *= q_over_p_plus_q;
            const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
            const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
            a0 *= theta;

            const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
            const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
            double R000_0 = 1;
            double R000_1 = -a0 / 3;
            if (boys_input > 1e-14) {
                const double sqrt_boys_input = sqrt(boys_input);
                R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
                R000_1 = -a0 * (R000_0 - exp(-boys_input)) / boys_input;
            }
            deri_dCx_per_pair += prefactor * R000_1 * PCx;
            deri_dCy_per_pair += prefactor * R000_1 * PCy;
            deri_dCz_per_pair += prefactor * R000_1 * PCz;
        }

        const double D = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair];
        deri_dCx_pair_sum += deri_dCx_per_pair * D;
        deri_dCy_pair_sum += deri_dCy_per_pair * D;
        deri_dCz_pair_sum += deri_dCz_per_pair * D;
    }
    atomicAdd(output + task_grid + 0 * ngrids, deri_dCx_pair_sum);
    atomicAdd(output + task_grid + 1 * ngrids, deri_dCy_pair_sum);
    atomicAdd(output + task_grid + 2 * ngrids, deri_dCz_pair_sum);
}

__global__
static void GINTfill_int3c1e_ip2_charge_contracted_kernel00(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                                            const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j, const int* gridslice,
                                                            const double omega, const double* grid_points, const double* charge_exponents)
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

    const double* grid_point = grid_points + task_grid * 4;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double deri_dCx = 0;
    double deri_dCy = 0;
    double deri_dCz = 0;
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
        const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
        const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
        a0 *= q_over_p_plus_q;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
        const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
        a0 *= theta;

        const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
        const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
        double R000_0 = 1;
        double R000_1 = -a0 / 3;
        if (boys_input > 1e-14) {
            const double sqrt_boys_input = sqrt(boys_input);
            R000_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
            R000_1 = -a0 * (R000_0 - exp(-boys_input)) / boys_input;
        }
        const double R100_0 = R000_1 * PCx;
        const double R010_0 = R000_1 * PCy;
        const double R001_0 = R000_1 * PCz;
        deri_dCx += prefactor * R100_0;
        deri_dCy += prefactor * R010_0;
        deri_dCz += prefactor * R001_0;
    }

    const double charge = grid_point[3];
    deri_dCx *= charge;
    deri_dCy *= charge;
    deri_dCz *= charge;

    const int i_atom = gridslice[task_grid];
    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    atomicAdd(output + (i0 + j0 * stride_j + 0 * stride_ij + i_atom * 3 * stride_ij), deri_dCx);
    atomicAdd(output + (i0 + j0 * stride_j + 1 * stride_ij + i_atom * 3 * stride_ij), deri_dCy);
    atomicAdd(output + (i0 + j0 * stride_j + 2 * stride_ij + i_atom * 3 * stride_ij), deri_dCz);
}
