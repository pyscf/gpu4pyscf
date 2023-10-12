/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2023 Qiming Sun
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

template<int NROOTS, int GOUTSIZE>
__global__
static void GINTint2e_get_veff_ip1_kernel(GINTEnvVars envs,
                                          JKMatrix jk,
                                          BasisProdOffsets offsets) {

  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao_squared = nao * nao;

  int i, j, k, l, f;

  double norm = envs.fac;
  double omega = envs.omega;
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;

  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double uw[NROOTS * 2];
  double local_cache[NROOTS * GPU_AO_LMAX + GOUTSIZE];
  memset(local_cache, 0, sizeof(double) * (NROOTS * GPU_AO_LMAX + GOUTSIZE));
  double * __restrict__ g = local_cache + NROOTS * GPU_AO_LMAX;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  if (vk == NULL) {
    if(vj == NULL) {
      return;
    } else {
      double d_ij, d_kl;
      double shell_ix = 0,
             shell_iy = 0,
             shell_iz = 0,
             shell_jx = 0,
             shell_jy = 0,
             shell_jz = 0;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = i_exponent[ij];
        double aj = j_exponent[ij];
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          double akl = a12[kl];
          double xkl = x12[kl];
          double ykl = y12[kl];
          double zkl = z12[kl];
          double xijxkl = xij - xkl;
          double yijykl = yij - ykl;
          double zijzkl = zij - zkl;
          double aijkl = aij + akl;
          double a1 = aij * akl;
          double a0 = a1 / aijkl;
          double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
          a0 *= theta;
          double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);

          GINTrys_root<NROOTS>(x, uw);
          GINTscale_u<NROOTS>(uw, theta);
          GINTg0_2e_2d4d_ip1<NROOTS>(envs, g, uw, norm,
                                 as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          dm = jk.dm;
          for(int i_dm = 0; i_dm < jk.n_dm; i_dm++) {
            for (f = 0, l = l0; l < l1; ++l) {
              for (k = k0; k < k1; ++k) {
                d_kl = dm[k + nao * l];
                for (j = j0; j < j1; ++j) {
                  for (i = i0; i < i1; ++i, ++f) {
                    d_ij = dm[i + nao * j];

                    GINTgout2e_ip1_per_function<NROOTS>(envs, g, ai, aj, f,
                                                        &s_ix, &s_iy, &s_iz,
                                                        &s_jx, &s_jy,
                                                        &s_jz);

                    double dm_component = d_kl * d_ij;

                    shell_ix += s_ix * dm_component;
                    shell_iy += s_iy * dm_component;
                    shell_iz += s_iz * dm_component;
                    shell_jx += s_jx * dm_component;
                    shell_jy += s_jy * dm_component;
                    shell_jz += s_jz * dm_component;
                  }
                }
              }
            }
            dm += nao_squared;
          }
        }
      }

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
  } else {
    double d_ik, d_il, d_jk, d_jl;
    if (vj == NULL) {
      double shell_ix = 0,
             shell_iy = 0,
             shell_iz = 0,
             shell_jx = 0,
             shell_jy = 0,
             shell_jz = 0;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = i_exponent[ij];
        double aj = j_exponent[ij];
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          double akl = a12[kl];
          double xkl = x12[kl];
          double ykl = y12[kl];
          double zkl = z12[kl];
          double xijxkl = xij - xkl;
          double yijykl = yij - ykl;
          double zijzkl = zij - zkl;
          double aijkl = aij + akl;
          double a1 = aij * akl;
          double a0 = a1 / aijkl;
          double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
          a0 *= theta;
          double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);

          GINTrys_root<NROOTS>(x, uw);
          GINTscale_u<NROOTS>(uw, theta);
          GINTg0_2e_2d4d_ip1<NROOTS>(envs, g, uw, norm,
                                 as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          dm = jk.dm;
          for(int i_dm = 0; i_dm < jk.n_dm; i_dm++) {
            for (f = 0, l = l0; l < l1; ++l) {
              for (k = k0; k < k1; ++k) {
                for (j = j0; j < j1; ++j) {
                  d_jl = dm[j + nao * l];
                  d_jk = dm[j + nao * k];
                  for (i = i0; i < i1; ++i, ++f) {
                    d_ik = dm[i + nao * k];
                    d_il = dm[i + nao * l];

                    GINTgout2e_ip1_per_function<NROOTS>(envs, g, ai, aj, f,
                                                        &s_ix, &s_iy, &s_iz,
                                                        &s_jx, &s_jy,
                                                        &s_jz);

                    double exchange_component = d_ik * d_jl + d_il * d_jk;

                    shell_ix += s_ix * exchange_component;
                    shell_iy += s_iy * exchange_component;
                    shell_iz += s_iz * exchange_component;
                    shell_jx += s_jx * exchange_component;
                    shell_jy += s_jy * exchange_component;
                    shell_jz += s_jz * exchange_component;
                  }
                }
              }
            }
            dm += nao_squared;
          }
        }
      }

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    } else {
      double d_ij, d_kl;
      double j_shell_ix = 0,
             j_shell_iy = 0,
             j_shell_iz = 0,
             j_shell_jx = 0,
             j_shell_jy = 0,
             j_shell_jz = 0;

      double k_shell_ix = 0,
             k_shell_iy = 0,
             k_shell_iz = 0,
             k_shell_jx = 0,
             k_shell_jy = 0,
             k_shell_jz = 0;


      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = i_exponent[ij];
        double aj = j_exponent[ij];
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          double akl = a12[kl];
          double xkl = x12[kl];
          double ykl = y12[kl];
          double zkl = z12[kl];
          double xijxkl = xij - xkl;
          double yijykl = yij - ykl;
          double zijzkl = zij - zkl;
          double aijkl = aij + akl;
          double a1 = aij * akl;
          double a0 = a1 / aijkl;
          double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
          a0 *= theta;
          double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);

          GINTrys_root<NROOTS>(x, uw);
          GINTscale_u<NROOTS>(uw, theta);
          GINTg0_2e_2d4d_ip1<NROOTS>(envs, g, uw, norm,
                                 as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          dm = jk.dm;
          for(int i_dm = 0; i_dm < jk.n_dm; i_dm++) {
            for (f = 0, l = l0; l < l1; ++l) {
              for (k = k0; k < k1; ++k) {
                d_kl = dm[k + nao * l];
                for (j = j0; j < j1; ++j) {
                  d_jl = dm[j + nao * l];
                  d_jk = dm[j + nao * k];
                  for (i = i0; i < i1; ++i, ++f) {
                    d_ij = dm[i + nao * j];
                    d_ik = dm[i + nao * k];
                    d_il = dm[i + nao * l];
                    GINTgout2e_ip1_per_function<NROOTS>(envs, g, ai, aj, f,
                                                        &s_ix, &s_iy, &s_iz,
                                                        &s_jx, &s_jy,
                                                        &s_jz);

                    double coulomb_component = d_ij * d_kl;
                    double exchange_component = d_ik * d_jl + d_il * d_jk;

                    j_shell_ix += s_ix * coulomb_component;
                    j_shell_iy += s_iy * coulomb_component;
                    j_shell_iz += s_iz * coulomb_component;
                    j_shell_jx += s_jx * coulomb_component;
                    j_shell_jy += s_jy * coulomb_component;
                    j_shell_jz += s_jz * coulomb_component;

                    k_shell_ix += s_ix * exchange_component;
                    k_shell_iy += s_iy * exchange_component;
                    k_shell_iz += s_iz * exchange_component;
                    k_shell_jx += s_jx * exchange_component;
                    k_shell_jy += s_jy * exchange_component;
                    k_shell_jz += s_jz * exchange_component;
                  }
                }
              }
            }
            dm += nao_squared;
          }
        }
      }

      atomicAdd(vj+ish*3  , j_shell_ix);
      atomicAdd(vj+ish*3+1, j_shell_iy);
      atomicAdd(vj+ish*3+2, j_shell_iz);
      atomicAdd(vj+jsh*3  , j_shell_jx);
      atomicAdd(vj+jsh*3+1, j_shell_jy);
      atomicAdd(vj+jsh*3+2, j_shell_jz);
      atomicAdd(vk+ish*3  , k_shell_ix);
      atomicAdd(vk+ish*3+1, k_shell_iy);
      atomicAdd(vk+ish*3+2, k_shell_iz);
      atomicAdd(vk+jsh*3  , k_shell_jx);
      atomicAdd(vk+jsh*3+1, k_shell_jy);
      atomicAdd(vk+jsh*3+2, k_shell_jz);
    }
  }


}


__global__
static void
GINTint2e_get_veff_ip1_kernel_0000(GINTEnvVars envs,
                                   JKMatrix jk,
                                   BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int * ao_loc = c_bpcache.ao_loc;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int i = ao_loc[ish];
  int j = ao_loc[jsh];
  int k = ao_loc[ksh];
  int l = ao_loc[lsh];

  int nbas = c_bpcache.nbas;
  double * __restrict__ bas_x = c_bpcache.bas_coords;
  double * __restrict__ bas_y = bas_x + nbas;
  double * __restrict__ bas_z = bas_y + nbas;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];

  double xj = bas_x[jsh];
  double yj = bas_y[jsh];
  double zj = bas_z[jsh];

  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ e12 = c_bpcache.e12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  double gout0 = 0, gout0_prime = 0;
  double gout1 = 0, gout1_prime = 0;
  double gout2 = 0, gout2_prime = 0;

  for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
    double ai = 2.0 * i_exponent[ij];
    double aj = 2.0 * j_exponent[ij];
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
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
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
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
      root0 /= root0 + 1 - root0 * theta;
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

      double g_1_prime = c00x_prime;
      double g_3_prime = c00y_prime;
      double g_5_prime = g_4 * c00z_prime;

      gout0 += g_1 * g_2 * g_4 * ai;
      gout1 += g_0 * g_3 * g_4 * ai;
      gout2 += g_0 * g_2 * g_5 * ai;

      gout0_prime += g_1_prime * g_2 * g_4 * aj;
      gout1_prime += g_0 * g_3_prime * g_4 * aj;
      gout2_prime += g_0 * g_2 * g_5_prime * aj;
    }
  }

  int nao = jk.nao;

  double * __restrict__ dm = jk.dm;
  double * __restrict__ vj = jk.vj;
  double * __restrict__ vk = jk.vk;

  if(vj != NULL) {
    double coulomb = dm[k + nao * l] * dm[i + nao * j];

    atomicAdd(vj+ish*3  , gout0       * coulomb);
    atomicAdd(vj+ish*3+1, gout1       * coulomb);
    atomicAdd(vj+ish*3+2, gout2       * coulomb);
    atomicAdd(vj+jsh*3  , gout0_prime * coulomb);
    atomicAdd(vj+jsh*3+1, gout1_prime * coulomb);
    atomicAdd(vj+jsh*3+2, gout2_prime * coulomb);
  }
  if (vk != NULL) {
    double exchange = dm[i + nao * k] * dm[j + nao * l]
                    + dm[i + nao * l] * dm[j + nao * k];

    atomicAdd(vk+ish*3  , gout0       * exchange);
    atomicAdd(vk+ish*3+1, gout1       * exchange);
    atomicAdd(vk+ish*3+2, gout2       * exchange);
    atomicAdd(vk+jsh*3  , gout0_prime * exchange);
    atomicAdd(vk+jsh*3+1, gout1_prime * exchange);
    atomicAdd(vk+jsh*3+2, gout2_prime * exchange);
  }


}