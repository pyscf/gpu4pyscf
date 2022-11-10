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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint/gint.h"
#include "gint/config.h"
#include "gint/cuda_alloc.cuh"
#include "gint/g2e.h"
#include "gint/cint2e.cuh"

typedef struct {
        int nao;
        int n_dm;
        double *dm;
        double *vj;
        double *vk;
} JKMatrix;

__constant__ GINTEnvVars c_envs;
__constant__ BasisProdCache c_bpcache;
__constant__ int16_t c_idx4c[NFffff*3];

#include "g2e.cu"
#include "contract_jk.cu"
#include "gint/rys_roots.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"

__host__
static int GINTrun_tasks_jk(JKMatrix *jk, BasisProdOffsets *offsets, GINTEnvVars *envs)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntask_kl < 65536*THREADSY);
    int type_ijkl;

    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    switch (nrys_roots) {
    case 1:
        //GINTint2e_jk_kernel<1, GOUTSIZE1> <<<blocks, threads>>>(*offsets);
        if (envs->nf == 1) {
            GINTint2e_jk_kernel0000<<<blocks, threads>>>(*jk, *offsets);
        } else {
            GINTint2e_jk_kernel1000<<<blocks, threads>>>(*jk, *offsets);
        }
        break;
    case 2:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (1<<6)|(0<<4)|(1<<2)|0: GINTint2e_jk_kernel1010<<<blocks, threads>>>(*jk, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|1: GINTint2e_jk_kernel1011<<<blocks, threads>>>(*jk, *offsets); break;
        case (1<<6)|(1<<4)|(0<<2)|0: GINTint2e_jk_kernel1100<<<blocks, threads>>>(*jk, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|0: GINTint2e_jk_kernel1110<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(0<<4)|(0<<2)|0: GINTint2e_jk_kernel2000<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|0: GINTint2e_jk_kernel2010<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(1<<4)|(0<<2)|0: GINTint2e_jk_kernel2100<<<blocks, threads>>>(*jk, *offsets); break;
        case (3<<6)|(0<<4)|(0<<2)|0: GINTint2e_jk_kernel3000<<<blocks, threads>>>(*jk, *offsets); break;
        default:
            fprintf(stderr, "roots=2 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 3:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (1<<6)|(1<<4)|(1<<2)|1: GINTint2e_jk_kernel1111<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|1: GINTint2e_jk_kernel2011<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|0: GINTint2e_jk_kernel2020<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|1: GINTint2e_jk_kernel2021<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|0: GINTint2e_jk_kernel2110<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|1: GINTint2e_jk_kernel2111<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(1<<4)|(2<<2)|0: GINTint2e_jk_kernel2120<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(2<<4)|(0<<2)|0: GINTint2e_jk_kernel2200<<<blocks, threads>>>(*jk, *offsets); break;
        case (2<<6)|(2<<4)|(1<<2)|0: GINTint2e_jk_kernel2210<<<blocks, threads>>>(*jk, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|0: GINTint2e_jk_kernel3010<<<blocks, threads>>>(*jk, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|1: GINTint2e_jk_kernel3011<<<blocks, threads>>>(*jk, *offsets); break;
        case (3<<6)|(0<<4)|(2<<2)|0: GINTint2e_jk_kernel3020<<<blocks, threads>>>(*jk, *offsets); break;
        case (3<<6)|(1<<4)|(0<<2)|0: GINTint2e_jk_kernel3100<<<blocks, threads>>>(*jk, *offsets); break;
        case (3<<6)|(1<<4)|(1<<2)|0: GINTint2e_jk_kernel3110<<<blocks, threads>>>(*jk, *offsets); break;
        case (3<<6)|(2<<4)|(0<<2)|0: GINTint2e_jk_kernel3200<<<blocks, threads>>>(*jk, *offsets); break;
        default:
            fprintf(stderr, "roots=3 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 4:
        GINTint2e_jk_kernel<4, GOUTSIZE4> <<<blocks, threads>>>(*jk, *offsets);
        break;
    case 5:
        GINTint2e_jk_kernel<5, GOUTSIZE5> <<<blocks, threads>>>(*jk, *offsets);
        break;
    case 6:
        GINTint2e_jk_kernel<6, GOUTSIZE6> <<<blocks, threads>>>(*jk, *offsets);
        break;
    case 7:
        GINTint2e_jk_kernel<7, GOUTSIZE7> <<<blocks, threads>>>(*jk, *offsets);
        break;
    //case 8:
    //    GINTint2e_jk_kernel<8, GOUTSIZE8> <<<blocks, threads>>>(*jk, *offsets);
    //    break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTint2e_jk_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" { __host__
int GINTbuild_jk(BasisProdCache *bpcache,
                 double *vj, double *vk, double *dm, int nao, int n_dm,
                 int *bins_locs_ij, int *bins_locs_kl, int nbins,
                 int cp_ij_id, int cp_kl_id)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    GINTinit_EnvVars(&envs, cp_ij, cp_kl);
    if (envs.nrys_roots >= 8) {
        return 2;
    }

    if (envs.nrys_roots > 2) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        int *idx_ij = (int *)malloc(sizeof(int) * envs.nfi * envs.nfj * 3);
        int *idx_kl = (int *)malloc(sizeof(int) * envs.nfk * envs.nfl * 3);
        GINTinit_2c_gidx(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
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
    int nroots2 = envs.nrys_roots * 2;
    int kl_bin, ij_bin1;
    double *uw_buf, *d_uw;
    size_t uw_size = 0;
    if (envs.nrys_roots > POLYFIT_ORDER) {
        for (kl_bin = 0; kl_bin < nbins; ++kl_bin) {
            ij_bin1 = nbins - kl_bin;
            int bas_ij0 = bins_locs_ij[0];
            int bas_ij1 = bins_locs_ij[ij_bin1];
            int bas_kl0 = bins_locs_kl[kl_bin];
            int bas_kl1 = bins_locs_kl[kl_bin+1];
            int ntasks_ij = bas_ij1 - bas_ij0;
            int ntasks_kl = bas_kl1 - bas_kl0;
            uw_size = MAX(uw_size, ntasks_ij * ntasks_kl);
        }
        uw_size *= envs.nprim_ij * envs.nprim_kl * nroots2;
        checkCudaErrors(cudaHostAlloc(&uw_buf, sizeof(double) * uw_size,
                                      cudaHostAllocMapped));
        checkCudaErrors(cudaMalloc(&d_uw, sizeof(double) * uw_size));
        envs.uw = d_uw;
    }

    assert(nao < 32768);
    envs.nao = nao;
    checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
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
    for (kl_bin = 0; kl_bin < nbins; kl_bin++) {
        int bas_kl0 = bins_locs_kl[kl_bin];
        int bas_kl1 = bins_locs_kl[kl_bin+1];
        int ntasks_kl = bas_kl1 - bas_kl0;
        if (ntasks_kl <= 0) {
            continue;
        }
        // ij_bin + kl_bin < nbins <~> e_ij*e_kl < cutoff
        ij_bin1 = nbins - kl_bin;
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

        if (envs.nrys_roots > POLYFIT_ORDER) {
            // move rys roots and weights to device
            GINTinit_uw_s2(uw_buf, &offsets, &envs, bpcache);
            uw_size = (size_t)ntasks_ij * ntasks_kl * envs.nprim_ij * envs.nprim_kl * nroots2;
            checkCudaErrors(cudaMemcpy(d_uw, uw_buf, sizeof(double) * uw_size,
                                       cudaMemcpyHostToDevice));
        }
        int err = GINTrun_tasks_jk(&jk, &offsets, &envs);
        if (err != 0) {
            return err;
        }
    }

    if (envs.nrys_roots > POLYFIT_ORDER) {
        checkCudaErrors(cudaFreeHost(uw_buf));
        FREE(d_uw);
    }
    if (envs.nf > NFffff) {
        FREE(envs.idx);
    }
    return 0;
}
}
