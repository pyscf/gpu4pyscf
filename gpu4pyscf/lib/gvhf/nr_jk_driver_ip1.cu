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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "gvhf.h"

#include "gint/gint.h"
#include "gint/config.h"
#include "gint/cuda_alloc.cuh"
#include "gint/g2e.h"
#include "gint/cint2e.cuh"
#include "gint/rys_roots.cu"
#include "contract_jk.cu"
#include "g2e_ip1.cu"
#include "g2e_ip1_root2.cu"
#include "g2e_ip1_root3.cu"


__host__
static int GINTrun_tasks_ip1_jk(JKMatrix *jk,
                                BasisProdOffsets *offsets,
                                GINTEnvVars *envs)
{
  int nrys_roots = envs->nrys_roots;
  int ntasks_ij = offsets->ntasks_ij;
  int ntasks_kl = offsets->ntasks_kl;
  assert(ntasks_kl < 65536*THREADSY);
  int type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;

  dim3 threads(THREADSX, THREADSY);
  dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
  switch (nrys_roots) {
    case 1:
      switch (type_ijkl) {
        case 0b0000: GINTint2e_ip1_jk_kernel_0000<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        default:
          fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
      }
      break;

    case 2:
      switch (type_ijkl) {
        case (0<<6)|(0<<4)|(1<<2)|0: GINTint2e_ip1_jk_kernel_0010<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (0<<6)|(0<<4)|(1<<2)|1: GINTint2e_ip1_jk_kernel_0011<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (0<<6)|(0<<4)|(2<<2)|0: GINTint2e_ip1_jk_kernel_0020<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(0<<4)|(0<<2)|0: GINTint2e_ip1_jk_kernel_1000<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|0: GINTint2e_ip1_jk_kernel_1010<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(1<<4)|(0<<2)|0: GINTint2e_ip1_jk_kernel_1100<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(0<<2)|0: GINTint2e_ip1_jk_kernel_2000<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        default:
          fprintf(stderr, "roots=2 type_ijkl %d\n", type_ijkl);
      }
      break;

    case 3:
      switch (type_ijkl) {
        case (0<<6)|(0<<4)|(2<<2)|1: GINTint2e_ip1_jk_kernel_0021<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (0<<6)|(0<<4)|(2<<2)|2: GINTint2e_ip1_jk_kernel_0022<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (0<<6)|(0<<4)|(3<<2)|0: GINTint2e_ip1_jk_kernel_0030<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (0<<6)|(0<<4)|(3<<2)|1: GINTint2e_ip1_jk_kernel_0031<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|1: GINTint2e_ip1_jk_kernel_1011<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(0<<4)|(2<<2)|0: GINTint2e_ip1_jk_kernel_1020<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(0<<4)|(2<<2)|1: GINTint2e_ip1_jk_kernel_1021<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(0<<4)|(3<<2)|0: GINTint2e_ip1_jk_kernel_1030<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|0: GINTint2e_ip1_jk_kernel_1110<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|1: GINTint2e_ip1_jk_kernel_1111<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(1<<4)|(2<<2)|0: GINTint2e_ip1_jk_kernel_1120<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|0: GINTint2e_ip1_jk_kernel_2010<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|1: GINTint2e_ip1_jk_kernel_2011<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|0: GINTint2e_ip1_jk_kernel_2020<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(1<<4)|(0<<2)|0: GINTint2e_ip1_jk_kernel_2100<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|0: GINTint2e_ip1_jk_kernel_2110<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(2<<4)|(0<<2)|0: GINTint2e_ip1_jk_kernel_2200<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(0<<4)|(0<<2)|0: GINTint2e_ip1_jk_kernel_3000<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|0: GINTint2e_ip1_jk_kernel_3010<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(1<<4)|(0<<2)|0: GINTint2e_ip1_jk_kernel_3100<<<blocks, threads>>>(*envs, *jk, *offsets); break;
        default:
          fprintf(stderr, "roots=3 type_ijkl %d\n", type_ijkl);
      }
      break;

    case 4:
      GINTint2e_ip1_jk_kernel<4, NABLAGOUTSIZE4> <<<blocks, threads>>>(*envs, *jk, *offsets);
      break;
    case 5:
      GINTint2e_ip1_jk_kernel<5, NABLAGOUTSIZE5> <<<blocks, threads>>>(*envs, *jk, *offsets);
      break;
    case 6:
      GINTint2e_ip1_jk_kernel<6, NABLAGOUTSIZE6> <<<blocks, threads>>>(*envs, *jk, *offsets);
      break;
    case 7:
      GINTint2e_ip1_jk_kernel<7, NABLAGOUTSIZE7> <<<blocks, threads>>>(*envs, *jk, *offsets);
      break;
    default:
      fprintf(stderr, "rys roots %d\n", nrys_roots);
      return 1;

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error of GINTint2e_ip1_jk_kernel: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}

extern "C" {__host__

int GINTbuild_ip1_jk(BasisProdCache *bpcache,
                     double *vj, double *vk, double *dm, int nao, int n_dm,
                     int *bins_locs_ij, int *bins_locs_kl,
                     double *bins_floor_ij, double *bins_floor_kl,
                     int nbins_ij, int nbins_kl,
                     int cp_ij_id, int cp_kl_id, double omega, double log_cutoff, double sub_dm_cond,
                     double *dm_sh, int nshls,
                     double *log_q_ij, double *log_q_kl)
{
  ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
  ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
  GINTEnvVars envs;
  int ng[4] = {0,0,0,0};
  GINTinit_EnvVars_nabla1i(&envs, cp_ij, cp_kl, ng);
  envs.omega = omega;
  if (envs.nrys_roots > POLYFIT_ORDER) {
      fprintf(stderr, "build_ip1_jk: unsupported rys order %d\n", envs.nrys_roots);
      return 2;
  }

  if (envs.nrys_roots > 2) {
    int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
    int *idx_ij = (int *)malloc(sizeof(int) * envs.nfi * envs.nfj * 3);
    int *idx_kl = (int *)malloc(sizeof(int) * envs.nfk * envs.nfl * 3);
    GINTinit_2c_gidx_nabla1i(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
    GINTinit_2c_gidx(idx_kl, cp_kl->l_bra, cp_kl->l_ket);
    GINTinit_4c_idx(idx4c, idx_ij, idx_kl, &envs);
    if (envs.nf > NFffff) {
      DEVICE_INIT(int16_t, d_idx4c, idx4c, envs.nf * 3);
      envs.idx = d_idx4c;
    } else {
      checkCudaErrors(cudaMemcpyToSymbol(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3));
    }
    free(idx4c);
    free(idx_ij);
    free(idx_kl);
  }

  // Data and buffers to be allocated on-device. Allocate them here to
  // reduce the calls to malloc
  int kl_bin, ij_bin1;
  assert(nao < 32768);
  envs.nao = nao;

//  checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
  // move bpcache to constant memory
  checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

  JKMatrix jk;
  jk.n_dm = n_dm;
  jk.nao = nao;
  jk.dm = dm;
  jk.vj = vj;
  jk.vk = vk;

  BasisProdOffsets offsets;
  int *bas_pairs_locs = bpcache->bas_pairs_locs;
  int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
  for (kl_bin = 0; kl_bin < nbins_kl; kl_bin++) {
    int bas_kl0 = bins_locs_kl[kl_bin];
    int bas_kl1 = bins_locs_kl[kl_bin+1];
    int ntasks_kl = bas_kl1 - bas_kl0;
    if (ntasks_kl <= 0) {
      continue;
    }
    // ij_bin + kl_bin < nbins <~> e_ij*e_kl < cutoff
    ij_bin1 = 0;
    double log_q_kl_bin, log_q_ij_bin;
    log_q_kl_bin = bins_floor_kl[kl_bin];
    for(int ij_bin = 0; ij_bin < nbins_ij; ij_bin++){
      log_q_ij_bin = bins_floor_ij[ij_bin];
      if (log_q_ij_bin + log_q_kl_bin < log_cutoff - sub_dm_cond){
        break;
      }
      ij_bin1++;
    }

    int bas_ij0 = bins_locs_ij[0];
    int bas_ij1 = bins_locs_ij[ij_bin1];
    int ntasks_ij = bas_ij1 - bas_ij0;
    if (ntasks_ij <= 0) {
      continue;
    }

    offsets.ntasks_ij = ntasks_ij;
    offsets.ntasks_kl = ntasks_kl;
    offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
    offsets.bas_kl = bas_pairs_locs[cp_kl_id] + bas_kl0;
    offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * envs.nprim_ij;
    offsets.primitive_kl = primitive_pairs_locs[cp_kl_id] + bas_kl0 * envs.nprim_kl;
    offsets.log_cutoff = log_cutoff;

    int err = GINTrun_tasks_ip1_jk(&jk, &offsets, &envs);
    if (err != 0) {
      return err;
    }
  }

  if (envs.nf > NFffff) {
    FREE(envs.idx);
  }
  return 0;

}
}
