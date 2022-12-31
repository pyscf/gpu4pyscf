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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "g2e.h"


void CINTcart_comp(int *nx, int *ny, int *nz, int lmax);
void CINTrys_roots(int nroots, double x, double *u, double *w);

void GINTinit_EnvVars(GINTEnvVars *envs,
                      ContractionProdType *cp_ij, ContractionProdType *cp_kl)
{
        int i_l = cp_ij->l_bra;
        int j_l = cp_ij->l_ket;
        int k_l = cp_kl->l_bra;
        int l_l = cp_kl->l_ket;
        int nfi = (i_l + 1) * (i_l + 2) / 2;
        int nfj = (j_l + 1) * (j_l + 2) / 2;
        int nfk = (k_l + 1) * (k_l + 2) / 2;
        int nfl = (l_l + 1) * (l_l + 2) / 2;
        int nroots = (i_l + j_l + k_l + l_l)/2 + 1;
        double fac = (M_PI*M_PI*M_PI)*2/SQRTPI;

        envs->i_l = i_l;
        envs->j_l = j_l;
        envs->k_l = k_l;
        envs->l_l = l_l;
        envs->nfi = nfi;
        envs->nfj = nfj;
        envs->nfk = nfk;
        envs->nfl = nfl;
        envs->nf = nfi * nfj * nfk * nfl;
        envs->nrys_roots = nroots;
        envs->fac = fac;

        int ibase = i_l >= j_l;
        int kbase = k_l >= l_l;
        envs->ibase = ibase;
        envs->kbase = kbase;

        int li1 = i_l + 1;
        int lj1 = j_l + 1;
        int lk1 = k_l + 1;
        int ll1 = l_l + 1;
        int di = nroots;
        int dj = di * li1;
        int dk = dj * lj1;
        int dl = dk * lk1;
        envs->g_size_ij = dk;
        envs->g_size    = dl * ll1;

        if (ibase) {
                envs->ijmin = j_l;
                envs->ijmax = i_l;
                envs->stride_ijmax = nroots;
                envs->stride_ijmin = nroots * li1;
        } else {
                envs->ijmin = i_l;
                envs->ijmax = j_l;
                envs->stride_ijmax = nroots;
                envs->stride_ijmin = nroots * lj1;
        }

        if (kbase) {
                envs->klmin = l_l;
                envs->klmax = k_l;
                envs->stride_klmax = dk;
                envs->stride_klmin = dk * lk1;
        } else {
                envs->klmin = k_l;
                envs->klmax = l_l;
                envs->stride_klmax = dk;
                envs->stride_klmin = dk * ll1;
        }

        envs->nprim_ij = cp_ij->nprim_12;
        envs->nprim_kl = cp_kl->nprim_12;
}

void GINTinit_2c_gidx(int *idx, int li, int lj)
{
        int nfi = (li + 1) * (li + 2) / 2;
        int nfj = (lj + 1) * (lj + 2) / 2;
        int nfij = nfi * nfj;
        int i, j, n;
        int stride_i, stride_j;
        int *idy = idx + nfij;
        int *idz = idx + nfij * 2;

        int i_nx[GPU_CART_MAX], i_ny[GPU_CART_MAX], i_nz[GPU_CART_MAX];
        int j_nx[GPU_CART_MAX], j_ny[GPU_CART_MAX], j_nz[GPU_CART_MAX];
        CINTcart_comp(i_nx, i_ny, i_nz, li);
        CINTcart_comp(j_nx, j_ny, j_nz, lj);

        if (li >= lj) {
                stride_i = 1;
                stride_j = li + 1;
        } else {
                stride_i = li + 1;
                stride_j = 1;
        }
        for (n = 0, j = 0; j < nfj; j++) {
        for (i = 0; i < nfi; i++, n++) {
                idx[n] = stride_j * j_nx[j] + stride_i * i_nx[i];
                idy[n] = stride_j * j_ny[j] + stride_i * i_ny[i];
                idz[n] = stride_j * j_nz[j] + stride_i * i_nz[i];
        } }
}

void GINTinit_4c_idx(int16_t *idx, int *ij_idx, int *kl_idx, GINTEnvVars *envs)
{
        int nfi = envs->nfi;
        int nfj = envs->nfj;
        int nfk = envs->nfk;
        int nfl = envs->nfl;
        int nfij = nfi * nfj;
        int nfkl = nfk * nfl;
        int nf = nfij * nfkl;
        int nroots = envs->nrys_roots;
        int g_size = envs->g_size;
        int g_size_ij = envs->g_size_ij;
        int *ij_idy = ij_idx + nfij;
        int *ij_idz = ij_idy + nfij;
        int *kl_idy = kl_idx + nfkl;
        int *kl_idz = kl_idy + nfkl;
        int16_t *idy = idx + nf;
        int16_t *idz = idx + nf * 2;
        int ofx = 0;
        int ofy = g_size;
        int ofz = g_size * 2;
        int n, ij, kl;
        for (n = 0, kl = 0; kl < nfkl; kl++) {
        for (ij = 0; ij < nfij; ij++, n++) {
                idx[n] = ofx + ij_idx[ij] * nroots + kl_idx[kl] * g_size_ij;
                idy[n] = ofy + ij_idy[ij] * nroots + kl_idy[kl] * g_size_ij;
                idz[n] = ofz + ij_idz[ij] * nroots + kl_idz[kl] * g_size_ij;
        } }
// TODO: copy to constant memory or global memory depends on the size of nf
}

void GINTinit_uw_s2(double *uw_buf, BasisProdOffsets *offsets,
                    GINTEnvVars *envs, BasisProdCache *bpcache)
{
        size_t ntasks_ij = offsets->ntasks_ij;
        size_t ntasks_kl = offsets->ntasks_kl;
        int nprim_ij = envs->nprim_ij;
        int nprim_kl = envs->nprim_kl;
        int nroots = envs->nrys_roots;
        int strides = envs->nprim_ij * envs->nprim_kl * nroots * 2;
        int n_primitive_pairs = bpcache->primitive_pairs_locs[bpcache->ncptype];
        double *a12 = bpcache->aexyz;
        double *x12 = bpcache->aexyz + n_primitive_pairs * 2;
        double *y12 = bpcache->aexyz + n_primitive_pairs * 3;
        double *z12 = bpcache->aexyz + n_primitive_pairs * 4;

#pragma omp parallel
{
        int ij, kl, task_ij, task_kl, bas_ij, bas_kl, prim_ij, prim_kl;
        size_t n;
        double *uw;
#pragma omp for schedule(static)
        for (n = 0; n < ntasks_ij*ntasks_kl; n++) {
                task_ij = n % ntasks_ij;
                task_kl = n / ntasks_ij;
                bas_ij = offsets->bas_ij + task_ij;
                bas_kl = offsets->bas_kl + task_kl;
                if (bas_ij < bas_kl) {
                        continue;
                }
                prim_ij = offsets->primitive_ij + task_ij * nprim_ij;
                prim_kl = offsets->primitive_kl + task_kl * nprim_kl;
                uw = uw_buf + n * strides;
                for (ij = prim_ij; ij < prim_ij+nprim_ij; ij++) {
                for (kl = prim_kl; kl < prim_kl+nprim_kl; kl++) {
                        double aij = a12[ij];
                        double xij = x12[ij];
                        double yij = y12[ij];
                        double zij = z12[ij];
                        double akl = a12[kl];
                        double xkl = x12[kl];
                        double ykl = y12[kl];
                        double zkl = z12[kl];
                        double rx = xij - xkl;
                        double ry = yij - ykl;
                        double rz = zij - zkl;
                        double aijkl = aij + akl;
                        double a0 = aij * akl / aijkl;
                        double x = a0 * (rx * rx + ry * ry + rz * rz);
                        double *u = uw;
                        double *w = uw + nroots;
                        CINTrys_roots(nroots, x, u, w);
                        uw += nroots * 2;
                } }
        }
}
}

void GINTinit_uw_s1(double *uw_buf, BasisProdOffsets *offsets,
                    GINTEnvVars *envs, BasisProdCache *bpcache)
{
        size_t ntasks_ij = offsets->ntasks_ij;
        size_t ntasks_kl = offsets->ntasks_kl;
        int nprim_ij = envs->nprim_ij;
        int nprim_kl = envs->nprim_kl;
        int nroots = envs->nrys_roots;
        int strides = envs->nprim_ij * envs->nprim_kl * nroots * 2;
        int n_primitive_pairs = bpcache->primitive_pairs_locs[bpcache->ncptype];
        double *a12 = bpcache->aexyz;
        double *x12 = bpcache->aexyz + n_primitive_pairs * 2;
        double *y12 = bpcache->aexyz + n_primitive_pairs * 3;
        double *z12 = bpcache->aexyz + n_primitive_pairs * 4;

#pragma omp parallel
{
        int ij, kl, task_ij, task_kl, prim_ij, prim_kl;
        size_t n;
        double *uw;
#pragma omp for schedule(static)
        for (n = 0; n < ntasks_ij*ntasks_kl; n++) {
                task_ij = n % ntasks_ij;
                task_kl = n / ntasks_ij;
                prim_ij = offsets->primitive_ij + task_ij * nprim_ij;
                prim_kl = offsets->primitive_kl + task_kl * nprim_kl;
                uw = uw_buf + n * strides;
                for (ij = prim_ij; ij < prim_ij+nprim_ij; ij++) {
                for (kl = prim_kl; kl < prim_kl+nprim_kl; kl++) {
                        double aij = a12[ij];
                        double xij = x12[ij];
                        double yij = y12[ij];
                        double zij = z12[ij];
                        double akl = a12[kl];
                        double xkl = x12[kl];
                        double ykl = y12[kl];
                        double zkl = z12[kl];
                        double rx = xij - xkl;
                        double ry = yij - ykl;
                        double rz = zij - zkl;
                        double aijkl = aij + akl;
                        double a0 = aij * akl / aijkl;
                        double x = a0 * (rx * rx + ry * ry + rz * rz);
                        double *u = uw;
                        double *w = uw + nroots;
                        CINTrys_roots(nroots, x, u, w);
                        uw += nroots * 2;
                } }
        }
}
}

void GINTinit_EnvVars_nabla1i(GINTEnvVars *envs,
                              ContractionProdType *cp_ij,
                              ContractionProdType *cp_kl)
{
  int i_l = cp_ij->l_bra;
  int j_l = cp_ij->l_ket;
  int k_l = cp_kl->l_bra;
  int l_l = cp_kl->l_ket;
  int nfi = (i_l + 1) * (i_l + 2) / 2;
  int nfj = (j_l + 1) * (j_l + 2) / 2;
  int nfk = (k_l + 1) * (k_l + 2) / 2;
  int nfl = (l_l + 1) * (l_l + 2) / 2;
  int nroots = (i_l + j_l + k_l + l_l + 1)/2 + 1;
  double fac = (M_PI*M_PI*M_PI)*2/SQRTPI;

  envs->i_l = i_l;
  envs->j_l = j_l;
  envs->k_l = k_l;
  envs->l_l = l_l;
  envs->nfi = nfi;
  envs->nfj = nfj;
  envs->nfk = nfk;
  envs->nfl = nfl;
  envs->nf = nfi * nfj * nfk * nfl;
  envs->nrys_roots = nroots;
  envs->fac = fac;

  int ibase = i_l >= j_l;
  int kbase = k_l >= l_l;
  envs->ibase = ibase;
  envs->kbase = kbase;

  int li1 = i_l + 2;
  int lj1 = j_l + 2;
  int lk1 = k_l + 1;
  int ll1 = l_l + 1;
  int di = nroots;
  int dj = di * li1;
  int dk = dj * lj1;
  int dl = dk * lk1;
  envs->g_size_ij = dk;
  envs->g_size    = dl * ll1;

  if (ibase) {
    envs->ijmin = j_l;
    envs->ijmax = i_l;
    envs->stride_ijmax = nroots;
    envs->stride_ijmin = nroots * li1;
  } else {
    envs->ijmin = i_l;
    envs->ijmax = j_l;
    envs->stride_ijmax = nroots;
    envs->stride_ijmin = nroots * lj1;
  }

  if (kbase) {
    envs->klmin = l_l;
    envs->klmax = k_l;
    envs->stride_klmax = dk;
    envs->stride_klmin = dk * lk1;
  } else {
    envs->klmin = k_l;
    envs->klmax = l_l;
    envs->stride_klmax = dk;
    envs->stride_klmin = dk * ll1;
  }

  envs->nprim_ij = cp_ij->nprim_12;
  envs->nprim_kl = cp_kl->nprim_12;
}

void GINTinit_2c_gidx_nabla1i(int *idx, int li, int lj)
{
  int nfi = (li + 1) * (li + 2) / 2;
  int nfj = (lj + 1) * (lj + 2) / 2;
  int nfij = nfi * nfj;
  int i, j, n;
  int stride_i, stride_j;
  int *idy = idx + nfij;
  int *idz = idx + nfij * 2;

  int i_nx[GPU_CART_MAX], i_ny[GPU_CART_MAX], i_nz[GPU_CART_MAX];
  int j_nx[GPU_CART_MAX], j_ny[GPU_CART_MAX], j_nz[GPU_CART_MAX];
  CINTcart_comp(i_nx, i_ny, i_nz, li);
  CINTcart_comp(j_nx, j_ny, j_nz, lj);

  if (li >= lj) {
    stride_i = 1;
    stride_j = li + 2;
  } else {
    stride_i = li + 2;
    stride_j = 1;
  }
  for (n = 0, j = 0; j < nfj; j++) {
    for (i = 0; i < nfi; i++, n++) {
      idx[n] = stride_j * j_nx[j] + stride_i * i_nx[i];
      idy[n] = stride_j * j_ny[j] + stride_i * i_ny[i];
      idz[n] = stride_j * j_nz[j] + stride_i * i_nz[i];
    } }
}