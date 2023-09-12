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


template<int NROOTS>
__device__
void GINTgout2e_ip1_per_function(GINTEnvVars envs, double * __restrict__ g,
                                 double ai, double aj, int i,
                                 double * s_ix, double * s_iy,
                                 double * s_iz,
                                 double * s_jx, double * s_jy,
                                 double * s_jz) {

  int di = envs.stride_ijmax;
  int dj = envs.stride_ijmin;

  int nf = envs.nf;
  int16_t * idx = c_idx4c;

  if (nf > NFffff) {
    idx = envs.idx;
  }

  int16_t * idy = idx + nf;
  int16_t * idz = idx + nf * 2;
  int n, ix, iy, iz,
      ij_index_for_ix, i_index_for_ix, j_index_for_ix,
      ij_index_for_iy, i_index_for_iy, j_index_for_iy,
      ij_index_for_iz, i_index_for_iz, j_index_for_iz;

  ix = idx[i];
  ij_index_for_ix = ix % envs.g_size_ij;
  i_index_for_ix = ij_index_for_ix % dj / di;
  j_index_for_ix = ij_index_for_ix / dj;
  iy = idy[i];
  ij_index_for_iy = iy % envs.g_size_ij;
  i_index_for_iy = ij_index_for_iy % dj / di;
  j_index_for_iy = ij_index_for_iy / dj;
  iz = idz[i];
  ij_index_for_iz = iz % envs.g_size_ij;
  i_index_for_iz = ij_index_for_iz % dj / di;
  j_index_for_iz = ij_index_for_iz / dj;

  double s_ix_local, s_iy_local, s_iz_local, s_jx_local, s_jy_local, s_jz_local;

#pragma unroll
  for (n = 0; n < NROOTS; ++n) {
    s_ix_local += -i_index_for_ix *
             g[ix + n - di] * g[iy + n] * g[iz + n]
             + 2.0 * ai * g[ix + n + di] * g[iy + n] * g[iz + n];
    s_iy_local += -i_index_for_iy *
             g[ix + n] * g[iy + n - di] * g[iz + n]
             + 2.0 * ai * g[ix + n] * g[iy + n + di] * g[iz + n];
    s_iz_local += -i_index_for_iz *
             g[ix + n] * g[iy + n] * g[iz + n - di]
             + 2.0 * ai * g[ix + n] * g[iy + n] * g[iz + n + di];
    s_jx_local += -j_index_for_ix *
             g[ix + n - dj] * g[iy + n] * g[iz + n]
             + 2.0 * aj * g[ix + n + dj] * g[iy + n] * g[iz + n];
    s_jy_local += -j_index_for_iy *
             g[ix + n] * g[iy + n - dj] * g[iz + n]
             + 2.0 * aj * g[ix + n] * g[iy + n + dj] * g[iz + n];
    s_jz_local += -j_index_for_iz *
             g[ix + n] * g[iy + n] * g[iz + n - dj]
             + 2.0 * aj * g[ix + n] * g[iy + n] * g[iz + n + dj];
  }

  *s_ix = s_ix_local;
  *s_iy = s_iy_local;
  *s_iz = s_iz_local;
  *s_jx = s_jx_local;
  *s_jy = s_jy_local;
  *s_jz = s_jz_local;
}

template<int NROOTS>
__device__
void GINTgout2e_ip1(GINTEnvVars envs, double * __restrict__ gout, double * __restrict__ g,
                        double ai, double aj) {
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  int i;

  int nf = envs.nf;

  for (i = 0; i < envs.nf; i++) {
    GINTgout2e_ip1_per_function<NROOTS>(envs, g, ai, aj, i,
                                            &s_ix, &s_iy, &s_iz,
                                            &s_jx, &s_jy, &s_jz);

    gout[i] += s_ix;
    gout[i + nf] += s_iy;
    gout[i + 2 * nf] += s_iz;
    gout[i + 3 * nf] += s_jx;
    gout[i + 4 * nf] += s_jy;
    gout[i + 5 * nf] += s_jz;
  }
}

__device__
void GINTkernel_ip1_getjk(GINTEnvVars envs, JKMatrix jk, double * __restrict__ gout,
                          int ish, int jsh, int ksh, int lsh) {
  int * ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  // int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nf = envs.nf;
  int nao = jk.nao;
  int nao2 = nao * nao;
  int i, j, k, l, n, i_dm;
  int ip, jp, kp, lp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;
  int n_dm = jk.n_dm;
  double * vj = jk.vj;
  double * vk = jk.vk;
  double * __restrict__ dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;
  if (vk == NULL) {
    if (vj == NULL) {
      return;
    }
    double * __restrict__ buf_ij = gout + 6 * envs.nf;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
      memset(buf_ij, 0, 6 * nfij * sizeof(double));
      double * __restrict__ pgout = gout;
      for (l = l0; l < l1; ++l) {
        for (k = k0; k < k1; ++k) {
          d_kl = dm[k + nao * l];
          for (n = 0, j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
              s_ix = pgout[n];
              s_iy = pgout[n + nf];
              s_iz = pgout[n + 2 * nf];
              s_jx = pgout[n + 3 * nf];
              s_jy = pgout[n + 4 * nf];
              s_jz = pgout[n + 5 * nf];
              buf_ij[n] += s_ix * d_kl;
              buf_ij[n + nfij] += s_iy * d_kl;
              buf_ij[n + 2 * nfij] += s_iz * d_kl;
              buf_ij[n + 3 * nfij] += s_jx * d_kl;
              buf_ij[n + 4 * nfij] += s_jy * d_kl;
              buf_ij[n + 5 * nfij] += s_jz * d_kl;
            }
          }
          pgout += nfij;
        }
      }
      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj + i + nao * j, buf_ij[n]);
          atomicAdd(vj + i + nao * j + nao2, buf_ij[n + nfij]);
          atomicAdd(vj + i + nao * j + 2 * nao2, buf_ij[n + 2 * nfij]);
          atomicAdd(vj + j + nao * i, buf_ij[n + 3 * nfij]);
          atomicAdd(vj + j + nao * i + nao2, buf_ij[n + 4 * nfij]);
          atomicAdd(vj + j + nao * i + 2 * nao2, buf_ij[n + 5 * nfij]);
        }
      }
      dm += nao2;
      vj += 3 * nao2;
    }
    return;
  }

  // vk != NULL
  double buf_i[30];
  double buf_j[30];

  if (vj != NULL) {
    double * __restrict__ buf_ij = gout + 6 * envs.nf;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
      memset(buf_ij, 0, 6 * nfij * sizeof(double));
      double * __restrict__ pgout = gout;
      for (l = l0; l < l1; ++l) {
        memset(buf_i, 0, 3 * nfi * sizeof(double));
        memset(buf_j, 0, 3 * nfj * sizeof(double));

        for (k = k0; k < k1; ++k) {
          d_kl = dm[k + nao * l];

          for (n = 0, j = j0; j < j1; ++j) {
            jp = j - j0;
            v_jl_x = 0;
            v_jl_y = 0;
            v_jl_z = 0;
            d_jk = dm[j + nao * k];
            for (i = i0; i < i1; ++i, ++n) {
              ip = i - i0;
              s_ix = pgout[n];
              s_iy = pgout[n + nf];
              s_iz = pgout[n + 2 * nf];
              s_jx = pgout[n + 3 * nf];
              s_jy = pgout[n + 4 * nf];
              s_jz = pgout[n + 5 * nf];
              d_ik = dm[i + nao * k];
              v_jl_x += s_jx * d_ik;
              v_jl_y += s_jy * d_ik;
              v_jl_z += s_jz * d_ik;
              buf_ij[n] += s_ix * d_kl;
              buf_ij[n + nfij] += s_iy * d_kl;
              buf_ij[n + 2 * nfij] += s_iz * d_kl;
              buf_ij[n + 3 * nfij] += s_jx * d_kl;
              buf_ij[n + 4 * nfij] += s_jy * d_kl;
              buf_ij[n + 5 * nfij] += s_jz * d_kl;

              buf_i[ip] += s_ix * d_jk;
              buf_i[(ip + nfi)] += s_iy * d_jk;
              buf_i[(ip + 2 * nfi)] += s_iz * d_jk;
            }
            buf_j[jp] += v_jl_x;
            buf_j[(jp + nfj)] += v_jl_y;
            buf_j[(jp + 2 * nfj)] += v_jl_z;
          }
          pgout += nfij;
        }
        for (ip = 0; ip < nfi; ++ip) {
          atomicAdd(vk + i0 + ip + nao * l, buf_i[ip]);
          atomicAdd(vk + i0 + ip + nao * l + nao2,
                    buf_i[(ip + nfi)]);
          atomicAdd(vk + i0 + ip + nao * l + 2 * nao2,
                    buf_i[(ip + 2 * nfi)]);
        }
        for (jp = 0; jp < nfj; ++jp) {
          atomicAdd(vk + j0 + jp + nao * l, buf_j[jp]);
          atomicAdd(vk + j0 + jp + nao * l + nao2,
                    buf_j[(jp + nfj)]);
          atomicAdd(vk + j0 + jp + nao * l + 2 * nao2,
                    buf_j[(jp + 2 * nfj)]);
        }
      }
      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj + i + nao * j, buf_ij[n]);
          atomicAdd(vj + i + nao * j + nao2, buf_ij[n + nfij]);
          atomicAdd(vj + i + nao * j + 2 * nao2, buf_ij[n + 2 * nfij]);
          atomicAdd(vj + j + nao * i, buf_ij[n + 3 * nfij]);
          atomicAdd(vj + j + nao * i + nao2, buf_ij[n + 4 * nfij]);
          atomicAdd(vj + j + nao * i + 2 * nao2, buf_ij[n + 5 * nfij]);
        }
      }
      dm += nao2;
      vj += 3 * nao2;
      vk += 3 * nao2;
    }

  } else {  // vj == NULL, vk != NULL
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
      for (n = 0, l = l0; l < l1; ++l) {
        memset(buf_i, 0, 3 * nfi * sizeof(double));
        memset(buf_j, 0, 3 * nfj * sizeof(double));

        for (k = k0; k < k1; ++k) {
          for (j = j0; j < j1; ++j) {
            jp = j - j0;
            v_jl_x = 0;
            v_jl_y = 0;
            v_jl_z = 0;
            d_jk = dm[j + nao * k];
            for (i = i0; i < i1; ++i, ++n) {
              ip = i - i0;
              s_ix = gout[n];
              s_iy = gout[n + nf];
              s_iz = gout[n + 2 * nf];
              s_jx = gout[n + 3 * nf];
              s_jy = gout[n + 4 * nf];
              s_jz = gout[n + 5 * nf];
              d_ik = dm[i + nao * k];
              v_jl_x += s_jx * d_ik;
              v_jl_y += s_jy * d_ik;
              v_jl_z += s_jz * d_ik;

              buf_i[ip] += s_ix * d_jk;
              buf_i[(ip + nfi)] += s_iy * d_jk;
              buf_i[(ip + 2 * nfi)] += s_iz * d_jk;
            }
            buf_j[jp] += v_jl_x;
            buf_j[(jp + nfj)] += v_jl_y;
            buf_j[(jp + 2 * nfj)] += v_jl_z;
          }
        }
        for (ip = 0; ip < nfi; ++ip) {
          atomicAdd(vk + i0 + ip + nao * l, buf_i[ip]);
          atomicAdd(vk + i0 + ip + nao * l + nao2,
                    buf_i[(ip + nfi)]);
          atomicAdd(vk + i0 + ip + nao * l + 2 * nao2,
                    buf_i[(ip + 2 * nfi)]);
        }
        for (jp = 0; jp < nfj; ++jp) {
          atomicAdd(vk + j0 + jp + nao * l, buf_j[jp]);
          atomicAdd(vk + j0 + jp + nao * l + nao2,
                    buf_j[(jp + nfj)]);
          atomicAdd(vk + j0 + jp + nao * l + 2 * nao2,
                    buf_j[(jp + 2 * nfj)]);
        }
      }
      dm += nao2;
      vk += 3 * nao2;
    }
  }


  // vj == NULL, vk != NULL
  vk = jk.vk;
  dm = jk.dm;
  for (i_dm = 0; i_dm < n_dm; ++i_dm) {
    for (k = k0; k < k1; ++k) {
      kp = k - k0;
      memset(buf_i, 0, 3 * nfi * sizeof(double));
      memset(buf_j, 0, 3 * nfj * sizeof(double));

      for (l = l0; l < l1; ++l) {
        lp = l - l0;
        n = nfij * (lp * nfk + kp);
        for (j = j0; j < j1; ++j) {
          jp = j - j0;
          v_jk_x = 0;
          v_jk_y = 0;
          v_jk_z = 0;
          d_jl = dm[j + nao * l];
          for (i = i0; i < i1; ++i, ++n) {
            ip = i - i0;
            s_ix = gout[n];
            s_iy = gout[n + nf];
            s_iz = gout[n + 2 * nf];
            s_jx = gout[n + 3 * nf];
            s_jy = gout[n + 4 * nf];
            s_jz = gout[n + 5 * nf];

            d_il = dm[i + nao * l];
            v_jk_x += s_jx * d_il;
            v_jk_y += s_jy * d_il;
            v_jk_z += s_jz * d_il;

            buf_i[ip] += s_ix * d_jl;
            buf_i[(ip + nfi)] += s_iy * d_jl;
            buf_i[(ip + 2 * nfi)] += s_iz * d_jl;
          }
          buf_j[jp] += v_jk_x;
          buf_j[(jp + nfj)] += v_jk_y;
          buf_j[(jp + 2 * nfj)] += v_jk_z;
        }
      }
      for (ip = 0; ip < nfi; ++ip) {
        atomicAdd(vk + i0 + ip + nao * k, buf_i[ip]);
        atomicAdd(vk + i0 + ip + nao * k + nao2,
                  buf_i[(ip + nfi)]);
        atomicAdd(vk + i0 + ip + nao * k + 2 * nao2,
                  buf_i[(ip + 2 * nfi)]);
      }
      for (jp = 0; jp < nfj; ++jp) {
        atomicAdd(vk + j0 + jp + nao * k, buf_j[jp]);
        atomicAdd(vk + j0 + jp + nao * k + nao2,
                  buf_j[(jp + nfj)]);
        atomicAdd(vk + j0 + jp + nao * k + 2 * nao2,
                  buf_j[(jp + 2 * nfj)]);
      }
    }
    dm += nao * nao;
    vk += 3 * nao * nao;
  }
}

template<int NROOTS, int GOUTSIZE>
__global__
static void GINTint2e_ip1_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets) {
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
  int nao2 = nao * nao;

  int i, j, k, l, n, f, i_dm;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = envs.fac;

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
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;


  int n_dm = jk.n_dm;
  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double uw[NROOTS * 2];
  double gout[GOUTSIZE];
  double * __restrict__ g =
      gout + (3 * nfik + 3 * nfjk + 3 * nfil + 3 * nfjl + 6 * nfij) * n_dm;

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

    if (vj == NULL) {
      return;
    }

    double * __restrict__ buf_ij = gout;
    memset(buf_ij, 0, 6 * nfij * n_dm * sizeof(double));


    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = i_exponent[ij];
      double aj = j_exponent[ij];

      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
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
        double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
        if constexpr(NROOTS==6) {
          GINTrys_root6(x, uw);
        } else {
          GINTrys_root7(x, uw);
        }

        GINTg0_2e_2d4d<NROOTS>(envs, g, uw, norm,
                               as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        buf_ij = gout;
        dm = jk.dm;
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
          for (f = 0, l = l0; l < l1; ++l) {
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (n = 0, j = j0; j < j1; ++j) {
                for (i = i0; i < i1; ++i, ++n, ++f) {
                  GINTgout2e_ip1_per_function<NROOTS>(envs, g, ai, aj, f,
                                                          &s_ix, &s_iy, &s_iz,
                                                          &s_jx, &s_jy,
                                                          &s_jz);
                  buf_ij[n] += s_ix * d_kl;
                  buf_ij[n + nfij] += s_iy * d_kl;
                  buf_ij[n + 2 * nfij] += s_iz * d_kl;
                  buf_ij[n + 3 * nfij] += s_jx * d_kl;
                  buf_ij[n + 4 * nfij] += s_jy * d_kl;
                  buf_ij[n + 5 * nfij] += s_jz * d_kl;
                }
              }
            }
          }
          dm += nao2;
          buf_ij += 6 * nfij;
        }
      }
    }
    buf_ij = gout;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj + i + nao * j, buf_ij[n]);
          atomicAdd(vj + i + nao * j + nao2, buf_ij[n + nfij]);
          atomicAdd(vj + i + nao * j + 2 * nao2, buf_ij[n + 2 * nfij]);
          atomicAdd(vj + j + nao * i, buf_ij[n + 3 * nfij]);
          atomicAdd(vj + j + nao * i + nao2, buf_ij[n + 4 * nfij]);
          atomicAdd(vj + j + nao * i + 2 * nao2, buf_ij[n + 5 * nfij]);
        }
      }
      vj += 3 * nao2;
      buf_ij += 6 * nfij;
    }
  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;
      double * __restrict__ buf_ik = gout;
      double * __restrict__ buf_jk = buf_ik + 3 * nfik * n_dm;
      double * __restrict__ buf_il = buf_jk + 3 * nfjk * n_dm;
      double * __restrict__ buf_jl = buf_il + 3 * nfil * n_dm;
      double * __restrict__ buf_ij = buf_jl + 3 * nfjl * n_dm;
      memset(gout, 0,
             (3 * nfik + 3 * nfjk + 3 * nfil + 3 * nfjl + 6 * nfij) * n_dm *
             sizeof(double));

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = i_exponent[ij];
        double aj = j_exponent[ij];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          double aij = a12[ij];
          double xij = x12[ij];
          double yij = y12[ij];
          double zij = z12[ij];
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
          double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
          if constexpr(NROOTS==6) {
            GINTrys_root6(x, uw);
          } else {
            GINTrys_root7(x, uw);
          }

          GINTg0_2e_2d4d<NROOTS>(envs, g, uw, norm,
                                 as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          dm = jk.dm;
          p_buf_ij = buf_ij;
          for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            p_buf_il = buf_il + 3 * i_dm * nfil;
            p_buf_jl = buf_jl + 3 * i_dm * nfjl;
            for (f = 0, l = l0; l < l1; ++l) {
              p_buf_ik = buf_ik + 3 * i_dm * nfik;
              p_buf_jk = buf_jk + 3 * i_dm * nfjk;
              for (k = k0; k < k1; ++k) {
                d_kl = dm[k + nao * l];
                for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                  d_jl = dm[j + nao * l];
                  d_jk = dm[j + nao * k];

                  v_jl_x = 0;
                  v_jl_y = 0;
                  v_jl_z = 0;
                  v_jk_x = 0;
                  v_jk_y = 0;
                  v_jk_z = 0;

                  for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                    d_il = dm[i + nao * l];
                    d_ik = dm[i + nao * k];

                    GINTgout2e_ip1_per_function<NROOTS>(envs, g, ai, aj, f,
                                                            &s_ix, &s_iy,
                                                            &s_iz,
                                                            &s_jx, &s_jy,
                                                            &s_jz);
                    p_buf_ij[n] += s_ix * d_kl;
                    p_buf_ij[n + nfij] += s_iy * d_kl;
                    p_buf_ij[n + 2 * nfij] += s_iz * d_kl;
                    p_buf_ij[n + 3 * nfij] += s_jx * d_kl;
                    p_buf_ij[n + 4 * nfij] += s_jy * d_kl;
                    p_buf_ij[n + 5 * nfij] += s_jz * d_kl;

                    p_buf_ik[ip] += s_ix * d_jl;
                    p_buf_ik[ip + nfik] += s_iy * d_jl;
                    p_buf_ik[ip + 2 * nfik] += s_iz * d_jl;

                    p_buf_il[ip] += s_ix * d_jk;
                    p_buf_il[ip + nfil] += s_iy * d_jk;
                    p_buf_il[ip + 2 * nfil] += s_iz * d_jk;

                    v_jl_x += s_jx * d_ik;
                    v_jl_y += s_jy * d_ik;
                    v_jl_z += s_jz * d_ik;

                    v_jk_x += s_jx * d_il;
                    v_jk_y += s_jy * d_il;
                    v_jk_z += s_jz * d_il;
                  }

                  p_buf_jl[jp] += v_jl_x;
                  p_buf_jl[jp + nfjl] += v_jl_y;
                  p_buf_jl[jp + 2 * nfjl] += v_jl_z;

                  p_buf_jk[jp] += v_jk_x;
                  p_buf_jk[jp + nfjk] += v_jk_y;
                  p_buf_jk[jp + 2 * nfjk] += v_jk_z;
                }

                p_buf_jk += nfj;
                p_buf_ik += nfi;
              }

              p_buf_il += nfi;
              p_buf_jl += nfj;
            }
            dm += nao2;
            p_buf_ij += 6 * nfij;
          }
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = buf_ij;
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        for (n = 0, j = j0; j < j1; ++j) {
          for (i = i0; i < i1; ++i, ++n) {
            atomicAdd(vj + i + nao * j, p_buf_ij[n]);
            atomicAdd(vj + i + nao * j + nao2, p_buf_ij[n + nfij]);
            atomicAdd(vj + i + nao * j + 2 * nao2, p_buf_ij[n + 2 * nfij]);
            atomicAdd(vj + j + nao * i, p_buf_ij[n + 3 * nfij]);
            atomicAdd(vj + j + nao * i + nao2, p_buf_ij[n + 4 * nfij]);
            atomicAdd(vj + j + nao * i + 2 * nao2, p_buf_ij[n + 5 * nfij]);
          }
        }

        for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
          for (i = i0; i < i1; ++i, ++ip) {
            atomicAdd(vk + i + nao * k, p_buf_ik[ip]);
            atomicAdd(vk + i + nao * k + nao2, p_buf_ik[ip + nfik]);
            atomicAdd(vk + i + nao * k + 2 * nao2, p_buf_ik[ip + 2 * nfik]);
          }

          for (j = j0; j < j1; ++j, ++jp) {
            atomicAdd(vk + j + nao * k, p_buf_jk[jp]);
            atomicAdd(vk + j + nao * k + nao2, p_buf_jk[jp + nfjk]);
            atomicAdd(vk + j + nao * k + 2 * nao2, p_buf_jk[jp + 2 * nfjk]);
          }
        }

        for (ip = 0, jp = 0, n = 0, l = l0; l < l1; ++l) {
          for (i = i0; i < i1; ++i, ++ip) {
            atomicAdd(vk + i + nao * l, p_buf_il[ip]);
            atomicAdd(vk + i + nao * l + nao2, p_buf_il[ip + nfil]);
            atomicAdd(vk + i + nao * l + 2 * nao2, p_buf_il[ip + 2 * nfil]);
          }

          for (j = j0; j < j1; ++j, ++jp) {
            atomicAdd(vk + j + nao * l, p_buf_jl[jp]);
            atomicAdd(vk + j + nao * l + nao2, p_buf_jl[jp + nfjl]);
            atomicAdd(vk + j + nao * l + 2 * nao2, p_buf_jl[jp + 2 * nfjl]);
          }
        }

        vj += 3 * nao2;
        vk += 3 * nao2;
        p_buf_il += 3 * nfil;
        p_buf_jl += 3 * nfjl;
        p_buf_ik += 3 * nfik;
        p_buf_jk += 3 * nfjk;
        p_buf_ij += 6 * nfij;
      }


    } else { // only vk required
      double * __restrict__ buf_ik = gout;
      double * __restrict__ buf_jk = buf_ik + 3 * nfik * n_dm;
      double * __restrict__ buf_il = buf_jk + 3 * nfjk * n_dm;
      double * __restrict__ buf_jl = buf_il + 3 * nfil * n_dm;

      memset(gout, 0,
             (3 * nfik + 3 * nfjk + 3 * nfil + 3 * nfjl) * n_dm *
             sizeof(double));

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = i_exponent[ij];
        double aj = j_exponent[ij];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          double aij = a12[ij];
          double xij = x12[ij];
          double yij = y12[ij];
          double zij = z12[ij];
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
          double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
          if constexpr(NROOTS==6) {
            GINTrys_root6(x, uw);
          } else {
            GINTrys_root7(x, uw);
          }

          GINTg0_2e_2d4d<NROOTS>(envs, g, uw, norm,
                                 as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          dm = jk.dm;
          for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            p_buf_il = buf_il + 3 * i_dm * nfil;
            p_buf_jl = buf_jl + 3 * i_dm * nfjl;
            for (f = 0, l = l0; l < l1; ++l) {
              p_buf_ik = buf_ik + 3 * i_dm * nfik;
              p_buf_jk = buf_jk + 3 * i_dm * nfjk;
              for (k = k0; k < k1; ++k) {
                for (jp = 0, j = j0; j < j1; ++j, ++jp) {
                  d_jl = dm[j + nao * l];
                  d_jk = dm[j + nao * k];

                  v_jl_x = 0;
                  v_jl_y = 0;
                  v_jl_z = 0;
                  v_jk_x = 0;
                  v_jk_y = 0;
                  v_jk_z = 0;

                  for (ip = 0, i = i0; i < i1; ++i, ++ip, ++f) {
                    d_il = dm[i + nao * l];
                    d_ik = dm[i + nao * k];

                    GINTgout2e_ip1_per_function<NROOTS>(envs, g, ai, aj, f,
                                                            &s_ix, &s_iy,
                                                            &s_iz,
                                                            &s_jx, &s_jy,
                                                            &s_jz);

                    p_buf_ik[ip] += s_ix * d_jl;
                    p_buf_ik[ip + nfik] += s_iy * d_jl;
                    p_buf_ik[ip + 2 * nfik] += s_iz * d_jl;

                    p_buf_il[ip] += s_ix * d_jk;
                    p_buf_il[ip + nfil] += s_iy * d_jk;
                    p_buf_il[ip + 2 * nfil] += s_iz * d_jk;

                    v_jl_x += s_jx * d_ik;
                    v_jl_y += s_jy * d_ik;
                    v_jl_z += s_jz * d_ik;

                    v_jk_x += s_jx * d_il;
                    v_jk_y += s_jy * d_il;
                    v_jk_z += s_jz * d_il;
                  }

                  p_buf_jl[jp] += v_jl_x;
                  p_buf_jl[jp + nfjl] += v_jl_y;
                  p_buf_jl[jp + 2 * nfjl] += v_jl_z;

                  p_buf_jk[jp] += v_jk_x;
                  p_buf_jk[jp + nfjk] += v_jk_y;
                  p_buf_jk[jp + 2 * nfjk] += v_jk_z;
                }

                p_buf_jk += nfj;
                p_buf_ik += nfi;
              }

              p_buf_il += nfi;
              p_buf_jl += nfj;
            }
            dm += nao2;
          }
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {

        for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
          for (i = i0; i < i1; ++i, ++ip) {
            atomicAdd(vk + i + nao * k, p_buf_ik[ip]);
            atomicAdd(vk + i + nao * k + nao2, p_buf_ik[ip + nfik]);
            atomicAdd(vk + i + nao * k + 2 * nao2, p_buf_ik[ip + 2 * nfik]);
          }

          for (j = j0; j < j1; ++j, ++jp) {
            atomicAdd(vk + j + nao * k, p_buf_jk[jp]);
            atomicAdd(vk + j + nao * k + nao2, p_buf_jk[jp + nfjk]);
            atomicAdd(vk + j + nao * k + 2 * nao2, p_buf_jk[jp + 2 * nfjk]);
          }
        }

        for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
          for (i = i0; i < i1; ++i, ++ip) {
            atomicAdd(vk + i + nao * l, p_buf_il[ip]);
            atomicAdd(vk + i + nao * l + nao2, p_buf_il[ip + nfil]);
            atomicAdd(vk + i + nao * l + 2 * nao2, p_buf_il[ip + 2 * nfil]);
          }

          for (j = j0; j < j1; ++j, ++jp) {
            atomicAdd(vk + j + nao * l, p_buf_jl[jp]);
            atomicAdd(vk + j + nao * l + nao2, p_buf_jl[jp + nfjl]);
            atomicAdd(vk + j + nao * l + 2 * nao2, p_buf_jl[jp + 2 * nfjl]);
          }
        }
        vk += 3 * nao2;
        p_buf_il += 3 * nfil;
        p_buf_jl += 3 * nfjl;
        p_buf_ik += 3 * nfik;
        p_buf_jk += 3 * nfjk;
      }
    }
  }

}

__global__
static void
GINTint2e_ip1_jk_kernel_0000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets) {
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
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];

//  if(ish == jsh) {
//    norm *= 0.5;
//  }

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

  int ij, kl, i_dm;
  double gout0 = 0, gout0_prime = 0;
  double gout1 = 0, gout1_prime = 0;
  double gout2 = 0, gout2_prime = 0;

  for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
    double ai = i_exponent[ij];
    double aj = j_exponent[ij];
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
    }
  }

  int n_dm = jk.n_dm;
  int nao = jk.nao;
  size_t nao2 = nao * nao;
  double * __restrict__ dm = jk.dm;
  double * __restrict__ vj = jk.vj;
  double * __restrict__ vk = jk.vk;
  double d_0;

  for (i_dm = 0; i_dm < n_dm; ++i_dm) {
    if (vj != NULL) {
      d_0 = dm[k0 + nao * l0];
      atomicAdd(vj + i0 + nao * j0, gout0 * d_0);
      atomicAdd(vj + i0 + nao * j0 + nao2, gout1 * d_0);
      atomicAdd(vj + i0 + nao * j0 + 2 * nao2, gout2 * d_0);
      atomicAdd(vj + nao * i0 + j0, gout0_prime * d_0);
      atomicAdd(vj + nao * i0 + j0 + nao2, gout1_prime * d_0);
      atomicAdd(vj + nao * i0 + j0 + 2 * nao2, gout2_prime * d_0);
      vj += 3 * nao2;
    }
    if (vk != NULL) {
      // ijkl, jk -> il
      d_0 = dm[j0 + nao * k0];
      atomicAdd(vk + i0 + nao * l0, gout0 * d_0);
      atomicAdd(vk + i0 + nao * l0 + nao2, gout1 * d_0);
      atomicAdd(vk + i0 + nao * l0 + 2 * nao2, gout2 * d_0);
      // ijkl, jl -> ik
      d_0 = dm[j0 + nao * l0];
      atomicAdd(vk + i0 + nao * k0, gout0 * d_0);
      atomicAdd(vk + i0 + nao * k0 + nao2, gout1 * d_0);
      atomicAdd(vk + i0 + nao * k0 + 2 * nao2, gout2 * d_0);
      // ijkl, ik -> jl
      d_0 = dm[i0 + nao * k0];
      atomicAdd(vk + j0 + nao * l0, gout0_prime * d_0);
      atomicAdd(vk + j0 + nao * l0 + nao2, gout1_prime * d_0);
      atomicAdd(vk + j0 + nao * l0 + 2 * nao2, gout2_prime * d_0);
      // ijkl, il -> jk
      d_0 = dm[i0 + nao * l0];
      atomicAdd(vk + j0 + nao * k0, gout0_prime * d_0);
      atomicAdd(vk + j0 + nao * k0 + nao2, gout1_prime * d_0);
      atomicAdd(vk + j0 + nao * k0 + 2 * nao2, gout2_prime * d_0);
      vk += 3 * nao2;
    }
    dm += nao2;
  }
}

#if POLYFIT_ORDER >= 4

template<>
__global__
void GINTint2e_ip1_jk_kernel<4, NABLAGOUTSIZE4>(GINTEnvVars envs, JKMatrix jk,
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

  double uw[8];
  double gout[NABLAGOUTSIZE4];
  double * g = gout + 6 * envs.nf;
  memset(gout, 0, 6 * envs.nf * sizeof(double));

  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
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
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      GINTrys_root4(x, uw);
      GINTg0_2e_2d4d<4>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
      GINTgout2e_ip1<4>(envs, gout, g, ai, aj);
    }
  }

  GINTkernel_ip1_getjk(envs, jk, gout, ish, jsh, ksh, lsh);
}

#endif

#if POLYFIT_ORDER >= 5

template<>
__global__
void GINTint2e_ip1_jk_kernel<5, NABLAGOUTSIZE5>(GINTEnvVars envs, JKMatrix jk,
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

  double uw[10];
  double gout[NABLAGOUTSIZE5];
  double * g = gout + 6 * envs.nf;
  memset(gout, 0, 6 * envs.nf * sizeof(double));

  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ a1 = c_bpcache.a1;
  double * __restrict__ a2 = c_bpcache.a2;

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
  for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
    double ai = a1[ij];
    double aj = a2[ij];
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
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      GINTrys_root5(x, uw);
      GINTg0_2e_2d4d<5>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
      GINTgout2e_ip1<5>(envs, gout, g, ai, aj);
    }
  }

  GINTkernel_ip1_getjk(envs, jk, gout, ish, jsh, ksh, lsh);
}

#endif
