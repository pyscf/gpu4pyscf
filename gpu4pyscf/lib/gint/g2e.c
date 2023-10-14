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

static void CINTcart_comp(int *nx, int *ny, int *nz, int lmax)
{
        int inc = 0;
        int lx, ly, lz;

        for (lx = lmax; lx >= 0; lx--) {
                for (ly = lmax - lx; ly >= 0; ly--) {
                        lz = lmax - lx - ly;
                        nx[inc] = lx;
                        ny[inc] = ly;
                        nz[inc] = lz;
                        inc++;
                }
        }
}

void GINTinit_EnvVars(GINTEnvVars *envs,
                      ContractionProdType *cp_ij, ContractionProdType *cp_kl, int *ng)
{
        int i_l = cp_ij->l_bra;
        int j_l = cp_ij->l_ket;
        int k_l = cp_kl->l_bra;
        int l_l = cp_kl->l_ket;
        int nfi = (i_l + 1) * (i_l + 2) / 2;
        int nfj = (j_l + 1) * (j_l + 2) / 2;
        int nfk = (k_l + 1) * (k_l + 2) / 2;
        int nfl = (l_l + 1) * (l_l + 2) / 2;
        int li_ceil = i_l + ng[0];
        int lj_ceil = j_l + ng[1];
        int lk_ceil = k_l + ng[2];
        int ll_ceil = l_l + ng[3];
        int nroots = (li_ceil + lj_ceil + lk_ceil + ll_ceil)/2 + 1;
        double fac = (M_PI*M_PI*M_PI)*2/SQRTPI;

        envs->i_l = i_l;
        envs->j_l = j_l;
        envs->k_l = k_l;
        envs->l_l = l_l;
        envs->li_ceil = li_ceil;
        envs->lj_ceil = lj_ceil;
        envs->lk_ceil = lk_ceil;
        envs->ll_ceil = ll_ceil;
        envs->nfi = nfi;
        envs->nfj = nfj;
        envs->nfk = nfk;
        envs->nfl = nfl;
        envs->nf = nfi * nfj * nfk * nfl;
        envs->nrys_roots = nroots;
        envs->fac = fac;

        int ibase = 1;//i_l >= j_l; //li_ceil >= lj_ceil;
        int kbase = 1;//k_l >= l_l; //lk_ceil >= ll_ceil;
        envs->ibase = ibase;
        envs->kbase = kbase;

        int li1 = li_ceil + 1;
        int lj1 = lj_ceil + 1;
        int lk1 = lk_ceil + 1;
        int ll1 = ll_ceil + 1;
        int di = nroots;
        int dj = di * li1;
        int dk = dj * lj1;
        int dl = dk * lk1;
        envs->g_size_ij = dk;
        envs->g_size    = dl * ll1;

        if (ibase) {
                envs->ijmin = lj_ceil;
                envs->ijmax = li_ceil;
                envs->stride_ijmax = nroots;
                envs->stride_ijmin = nroots * li1;
                envs->stride_i = nroots;
                envs->stride_j = nroots * li1;
        } else {
                envs->ijmin = li_ceil;
                envs->ijmax = lj_ceil;
                envs->stride_ijmax = nroots;
                envs->stride_ijmin = nroots * lj1;
                envs->stride_i = nroots * lj1;
                envs->stride_j = nroots;
        }

        if (kbase) {
                envs->klmin = ll_ceil;
                envs->klmax = lk_ceil;
                envs->stride_klmax = dk;
                envs->stride_klmin = dk * lk1;
                envs->stride_k = dk;
                envs->stride_l = dk * lk1;
        } else {
                envs->klmin = lk_ceil;
                envs->klmax = ll_ceil;
                envs->stride_klmax = dk;
                envs->stride_klmin = dk * ll1;
                envs->stride_k = dk * ll1;
                envs->stride_l = dk;
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

void GINTg2e_index_xyz(int16_t *idx, GINTEnvVars *envs)
{
        const int16_t nfi = envs->nfi;
        const int16_t nfj = envs->nfj;
        const int16_t nfk = envs->nfk;
        const int16_t nfl = envs->nfl;
        const int16_t nf = envs->nf;
        int16_t i_l = envs->i_l;
        int16_t j_l = envs->j_l;
        int16_t k_l = envs->k_l;
        int16_t l_l = envs->l_l;
        int16_t di = envs->stride_i;
        int16_t dj = envs->stride_j;
        int16_t dk = envs->stride_k;
        int16_t dl = envs->stride_l;
        int16_t i, j, k, l, n;
        int16_t ofx, oflx, ofkx, ofjx;
        int16_t ofy, ofly, ofky, ofjy;
        int16_t ofz, oflz, ofkz, ofjz;
        int16_t *idy = idx + nf;
        int16_t *idz = idx + nf * 2;
        int i_nx[GPU_CART_MAX], i_ny[GPU_CART_MAX], i_nz[GPU_CART_MAX];
        int j_nx[GPU_CART_MAX], j_ny[GPU_CART_MAX], j_nz[GPU_CART_MAX];
        int k_nx[GPU_CART_MAX], k_ny[GPU_CART_MAX], k_nz[GPU_CART_MAX];
        int l_nx[GPU_CART_MAX], l_ny[GPU_CART_MAX], l_nz[GPU_CART_MAX];

        //if(!envs->ibase) { di = envs->stride_ijmin; dj = envs->stride_ijmax;}//i_l = envs->j_l;  j_l = envs->i_l; }
        //if(!envs->kbase) { dk = envs->stride_klmin; dl = envs->stride_klmax;}//k_l = envs->l_l;  l_l = envs->k_l; }

        CINTcart_comp(i_nx, i_ny, i_nz, i_l);
        CINTcart_comp(j_nx, j_ny, j_nz, j_l);
        CINTcart_comp(k_nx, k_ny, k_nz, k_l);
        CINTcart_comp(l_nx, l_ny, l_nz, l_l);

        ofx = 0;
        ofy = envs->g_size;
        ofz = envs->g_size * 2;
        n = 0;

        for (l = 0; l < nfl; l++){
                oflx = ofx + dl * l_nx[l];
                ofly = ofy + dl * l_ny[l];
                oflz = ofz + dl * l_nz[l];
                for (k = 0; k < nfk; k++){
                        ofkx = oflx + dk * k_nx[k];
                        ofky = ofly + dk * k_ny[k];
                        ofkz = oflz + dk * k_nz[k];
                        for (j = 0; j < nfj; j++){
                                ofjx = ofkx + dj * j_nx[j];
                                ofjy = ofky + dj * j_ny[j];
                                ofjz = ofkz + dj * j_nz[j];
                                /*
                                for (i = 0; i < nfi; i++, n++) {
                                        idx[n] = ofjx + di * i_nx[i];
                                        idy[n] = ofjy + di * i_ny[i];
                                        idz[n] = ofjz + di * i_nz[i];
                                }
                                */
                                switch (i_l){
                                        case 0:
                                                idx[n] = ofjx;
                                                idy[n] = ofjy;
                                                idz[n] = ofjz;
                                                n++;
                                                break;
                                        case 1:
                                                idx[n] = ofjx + di;
                                                idy[n] = ofjy;
                                                idz[n] = ofjz;
                                                n++;
                                                idx[n] = ofjx;
                                                idy[n] = ofjy + di;
                                                idz[n] = ofjz;
                                                n++;
                                                idx[n] = ofjx;
                                                idy[n] = ofjy;
                                                idz[n] = ofjz + di;
                                                n++;
                                                break;
                                        case 2:
                                                idx[n] = ofjx + di*2;
                                                idy[n] = ofjy;
                                                idz[n] = ofjz;
                                                n++;
                                                idx[n] = ofjx + di;
                                                idy[n] = ofjy + di;
                                                idz[n] = ofjz;
                                                n++;
                                                idx[n] = ofjx + di;
                                                idy[n] = ofjy;
                                                idz[n] = ofjz + di;
                                                n++;
                                                idx[n] = ofjx;
                                                idy[n] = ofjy + di*2;
                                                idz[n] = ofjz;
                                                n++;
                                                idx[n] = ofjx;
                                                idy[n] = ofjy + di;
                                                idz[n] = ofjz + di;
                                                n++;
                                                idx[n] = ofjx;
                                                idy[n] = ofjy;
                                                idz[n] = ofjz + di*2;
                                                n++;
                                                break;
                                        default:
                                                for (i = 0; i < nfi; i++, n++) {
                                                        idx[n] = ofjx + di * i_nx[i];
                                                        idy[n] = ofjy + di * i_ny[i];
                                                        idz[n] = ofjz + di * i_nz[i];
                                                }
                                                break;
                                }
                        }
                }
        }
}

void GINTinit_index1d_xyz(int *idx, int *l_locs)
{
        int *idx_x = idx;
        int *idx_y = idx + TOT_NF;
        int *idx_z = idx + 2 * TOT_NF;

        int n = 0;
        l_locs[0] = 0;
        for (int l = 0; l < GPU_LMAX+1; l++)
        {
                CINTcart_comp(idx_x+n, idx_y+n, idx_z+n, l);
                n += (l+1)*(l+2)/2;
                l_locs[l+1] = n;
        }
}

void GINTinit_EnvVars_nabla1i(GINTEnvVars *envs,
                              ContractionProdType *cp_ij,
                              ContractionProdType *cp_kl,
                              int *ng)
{
  int i_l = cp_ij->l_bra;
  int j_l = cp_ij->l_ket;
  int k_l = cp_kl->l_bra;
  int l_l = cp_kl->l_ket;
  int nfi = (i_l + 1) * (i_l + 2) / 2;
  int nfj = (j_l + 1) * (j_l + 2) / 2;
  int nfk = (k_l + 1) * (k_l + 2) / 2;
  int nfl = (l_l + 1) * (l_l + 2) / 2;
  int li_ceil = i_l + ng[0];
  int lj_ceil = j_l + ng[1];
  int lk_ceil = k_l + ng[2];
  int ll_ceil = l_l + ng[3];
  int nroots = (li_ceil + lj_ceil + lk_ceil + ll_ceil + 1)/2 + 1;
  double fac = (M_PI*M_PI*M_PI)*2/SQRTPI;

  envs->i_l = i_l;
  envs->j_l = j_l;
  envs->k_l = k_l;
  envs->l_l = l_l;
  envs->li_ceil = li_ceil;
  envs->lj_ceil = lj_ceil;
  envs->lk_ceil = lk_ceil;
  envs->ll_ceil = ll_ceil;
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

  int li1 = li_ceil + 2;
  int lj1 = lj_ceil + 2;
  int lk1 = lk_ceil + 1;
  int ll1 = ll_ceil + 1;
  int di = nroots;
  int dj = di * li1;
  int dk = dj * lj1;
  int dl = dk * lk1;
  envs->g_size_ij = dk;
  envs->g_size    = dl * ll1;

  if (ibase) {
    envs->ijmin = lj_ceil;
    envs->ijmax = li_ceil;
    envs->stride_ijmax = nroots;
    envs->stride_ijmin = nroots * li1;
    envs->stride_i = nroots;
    envs->stride_j = nroots * li1;
  } else {
    envs->ijmin = li_ceil;
    envs->ijmax = lj_ceil;
    envs->stride_ijmax = nroots;
    envs->stride_ijmin = nroots * lj1;
    envs->stride_i = nroots * lj1;
    envs->stride_j = nroots;
  }

  if (kbase) {
    envs->klmin = ll_ceil;
    envs->klmax = lk_ceil;
    envs->stride_klmax = dk;
    envs->stride_klmin = dk * lk1;
    envs->stride_k = dk;
    envs->stride_l = dk * lk1;
  } else {
    envs->klmin = lk_ceil;
    envs->klmax = ll_ceil;
    envs->stride_klmax = dk;
    envs->stride_klmin = dk * ll1;
    envs->stride_k = dk * ll1;
    envs->stride_l = dk;
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
