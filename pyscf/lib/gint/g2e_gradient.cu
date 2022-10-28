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

#include "nr_fill_ao_ints.cuh"

#include "g2e.h"
#include "gout2e.cuh"
#include "cint2e.cuh"
#include "rys_roots.cuh"
#include "fill_ints.cuh"

#define POLYFIT_ORDER   5
#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096

__global__
static void GINTfill_nabla1i_int2e_kernel0000(ERITensor eri,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = c_envs.fac;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  int ij, kl;

  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];

  double xj = bas_x[jsh];
  double yj = bas_y[jsh];
  double zj = bas_z[jsh];

  double gout0 = 0, gout0_prime = 0;
  double gout1 = 0, gout1_prime = 0;
  double gout2 = 0, gout2_prime = 0;

  int nprim_j =
      c_bpcache.primitive_functions_offsets[jsh + 1]
      - c_bpcache.primitive_functions_offsets[jsh];

  double * exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];
  double * exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
      double ai = exponent_i[(ij-prim_ij) / nprim_j];
      double aj = exponent_j[(ij-prim_ij) % nprim_j];
      double aij = a12[ij];
      double eij = e12[ij];
      double xij = x12[ij];
      double yij = y12[ij];
      double zij = z12[ij];
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;

      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl / (sqrt(aijkl) * a1);
      double root0, weight0;

      if (x < 3.e-7) {
        root0 = 0.5;
        weight0 = 1.;
      } else {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        weight0 = fmt0;
        double e = exp(-x);
        double b = .5 / x;
        double fmt1 = b * (fmt0 - e);
        root0 = fmt1 / (fmt0 - fmt1);
      }

      double u2 = a0 * root0;
      double tmp2 = akl * u2 / (u2 * aijkl + a1);
      double c00x = xij - xi - tmp2 * xijxkl;
      double c00y = yij - yi - tmp2 * yijykl;
      double c00z = zij - zi - tmp2 * zijzkl;

      double c00x_prime = xij - xj - tmp2 * xijxkl;
      double c00y_prime = yij - yj - tmp2 * yijykl;
      double c00z_prime = zij - zj - tmp2 * zijzkl;

      double g_0 = 1;
      double g_1 = c00x;
      double g_2 = 1;
      double g_3 = c00y;
      double g_4 = fac * weight0;
      double g_5 = g_4 * c00z;
      double g_6 = 2.0 * ai;

      double g_1_prime = c00x_prime;
      double g_3_prime = c00y_prime;
      double g_5_prime = g_4 * c00z_prime;
      double g_6_prime = 2.0 * aj;

      gout0 += g_1 * g_2 * g_4 * g_6;
      gout1 += g_0 * g_3 * g_4 * g_6;
      gout2 += g_0 * g_2 * g_5 * g_6;

      gout0_prime += g_1_prime * g_2 * g_4 * g_6_prime;
      gout1_prime += g_0 * g_3_prime * g_4 * g_6_prime;
      gout2_prime += g_0 * g_2 * g_5_prime * g_6_prime;
    } }

  int jstride = eri.stride_j;
  int kstride = eri.stride_k;
  int lstride = eri.stride_l;
  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh] - eri.ao_offsets_k;
  int l0 = ao_loc[lsh] - eri.ao_offsets_l;
  double* __restrict__ eri_ij = eri.data + l0*lstride+k0*kstride+j0*jstride+i0;
  double* __restrict__ eri_ji = eri.data + l0*lstride+k0*kstride+i0*jstride+j0;
  double* __restrict__ eri_ij_lk = eri.data + k0*lstride+l0*kstride+j0*jstride+i0;
  double* __restrict__ eri_ji_lk = eri.data + k0*lstride+l0*kstride+i0*jstride+j0;

  size_t xyz_stride = eri.n_elem;

  eri_ij[0] = gout0;
  eri_ij[1 * xyz_stride] = gout1;
  eri_ij[2 * xyz_stride] = gout2;
  eri_ij_lk[0] = gout0;
  eri_ij_lk[1 * xyz_stride] = gout1;
  eri_ij_lk[2 * xyz_stride] = gout2;

  eri_ji[0] = gout0_prime;
  eri_ji[1 * xyz_stride] = gout1_prime;
  eri_ji[2 * xyz_stride] = gout2_prime;
  eri_ji_lk[0] = gout0_prime;
  eri_ji_lk[1 * xyz_stride] = gout1_prime;
  eri_ji_lk[2 * xyz_stride] = gout2_prime;
}
