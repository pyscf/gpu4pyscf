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

#include "contract_jk.cu"
#include "gint/rys_roots.cu"

#include "g2e.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"

template <typename FloatType>
__host__
static int GINTrun_tasks_jk(JKMatrix *jk, JKMatrixMixedPrecision<FloatType> *jk_test, const BasisProdOffsets *offsets, const GINTEnvVars *envs)
{
    const int nrys_roots = envs->nrys_roots;
    const int ntasks_ij = offsets->ntasks_ij;
    const int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536 * THREADSY);
    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);

    int type_ijkl;
    switch (nrys_roots) {
    case 1:
        if (envs->nf == 1) {
            GINTint2e_jk_kernel0000<<<blocks, threads, 0>>>(*envs, *jk, *offsets);
        } else {
            GINTint2e_jk_kernel1000<<<blocks, threads, 0>>>(*envs, *jk, *offsets);
        }
        break;
    case 2:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (1<<6)|(0<<4)|(1<<2)|0: GINTint2e_jk_kernel1010<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|1: GINTint2e_jk_kernel1011<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(1<<4)|(0<<2)|0: GINTint2e_jk_kernel1100<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|0: GINTint2e_jk_kernel1110<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(0<<2)|0: GINTint2e_jk_kernel2000<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|0: GINTint2e_jk_kernel2010<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(1<<4)|(0<<2)|0: GINTint2e_jk_kernel2100<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(0<<4)|(0<<2)|0: GINTint2e_jk_kernel3000<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        default:
            GINTint2e_jk_kernel_general<2, GSIZE2, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
        }
        break;
    case 3:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (1<<6)|(1<<4)|(1<<2)|1: GINTint2e_jk_kernel1111<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|1: GINTint2e_jk_kernel2011<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|0: GINTint2e_jk_kernel2020<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|1: GINTint2e_jk_kernel2021<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|0: GINTint2e_jk_kernel2110<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|1: GINTint2e_jk_kernel2111<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(1<<4)|(2<<2)|0: GINTint2e_jk_kernel2120<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(2<<4)|(0<<2)|0: GINTint2e_jk_kernel2200<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (2<<6)|(2<<4)|(1<<2)|0: GINTint2e_jk_kernel2210<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|0: GINTint2e_jk_kernel3010<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|1: GINTint2e_jk_kernel3011<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(0<<4)|(2<<2)|0: GINTint2e_jk_kernel3020<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(1<<4)|(0<<2)|0: GINTint2e_jk_kernel3100<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(1<<4)|(1<<2)|0: GINTint2e_jk_kernel3110<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        case (3<<6)|(2<<4)|(0<<2)|0: GINTint2e_jk_kernel3200<<<blocks, threads, 0>>>(*envs, *jk, *offsets); break;
        default:
            GINTint2e_jk_kernel_general<3, GSIZE3, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
        }
        break;
    case 4:
        GINTint2e_jk_kernel_general<4, GSIZE4, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
    case 5:
        GINTint2e_jk_kernel_general<5, GSIZE5, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
    case 6:
        GINTint2e_jk_kernel_general<6, GSIZE6, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
    case 7:
        GINTint2e_jk_kernel_general<7, GSIZE7, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
    case 8:
        GINTint2e_jk_kernel_general<8, GSIZE8, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
    case 9:
        GINTint2e_jk_kernel_general<9, GSIZE9, FloatType> <<<blocks, threads, 0>>>(*envs, *jk_test, *offsets); break;
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
int GINTbuild_jk(const BasisProdCache *bpcache,
                 const BasisProductCacheSinglePrecision *bpcache_single, const BasisProductCacheDoublePrecision *bpcache_double,
                 double *vj, double *vk, double *dm, // miss a const here
                 float *vj_single_precision, float *vk_single_precision, const float *dm_single_precision,
                 const int nao, const int n_dm,
                 const int *bins_locs_ij, const int *bins_locs_kl,
                 const double *bins_floor_ij, const double *bins_floor_kl,
                 const int nbins_ij, const int nbins_kl,
                 const int cp_ij_id, const int cp_kl_id, // pair type id
                 const double omega,
                 const double log_cutoff, // overall cutoff
                 const double log_single_double_precision_threshold,
                 const double sub_dm_cond, // max element of density matrix
                 double *dm_sh, // unused, miss a const here
                 const int nshls, // unused
                 const double *log_q_ij, const double *log_q_kl // Schwarz upperbound
                 )
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    int ng[4] = {0,0,0,0};
    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > POLYFIT_ORDER) {
        fprintf(stderr, "build_jk: unsupported rys order %d\n", envs.nrys_roots);
        return 2;
    }

    if (envs.nrys_roots >= 2) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        int *idx_ij = (int *)malloc(sizeof(int) * envs.nfi * envs.nfj * 3);
        int *idx_kl = (int *)malloc(sizeof(int) * envs.nfk * envs.nfl * 3);
        GINTinit_2c_gidx(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
        GINTinit_2c_gidx(idx_kl, cp_kl->l_bra, cp_kl->l_ket);
        GINTinit_4c_idx(idx4c, idx_ij, idx_kl, &envs);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error of GINTbuild_int2e_kernel: %s\n", cudaGetErrorString(err));
            return 1;
        }

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
    assert(nao < 32768);
    envs.nao = nao;
    //checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
    // move bpcache to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache_single, bpcache_single, sizeof(BasisProductCacheSinglePrecision)));
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache_double, bpcache_double, sizeof(BasisProductCacheDoublePrecision)));

    JKMatrix jk; // Should be replaced by JKMatrixMixedPrecision in later code
    jk.n_dm = n_dm;
    jk.nao = nao;
    jk.dm = dm;
    jk.vj = vj;
    jk.vk = vk;
    jk.dm_sh = dm_sh;
    jk.nshls = nshls;
    BasisProdOffsets offsets;

    int n_tasks_count_single = 0;
    int n_tasks_count_double = 0;

    const int *bas_pairs_locs = bpcache->bas_pairs_locs;
    const int *primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int kl_bin = 0; kl_bin < nbins_kl; kl_bin++) {
        const int bas_kl0 = bins_locs_kl[kl_bin];
        const int bas_kl1 = bins_locs_kl[kl_bin+1];
        const int ntasks_kl = bas_kl1 - bas_kl0;
        if (ntasks_kl <= 0) {
            continue;
        }

        // ij_bin1 is the index of first bin out of cutoff
        int ij_bin_double_first = 0;
        int ij_bin_single_first = 0;
        const double log_q_kl_bin = bins_floor_kl[kl_bin];
        for (int ij_bin = 0; ij_bin < nbins_ij; ij_bin++) {
            const double log_q_ij_bin = bins_floor_ij[ij_bin];
            if (log_q_ij_bin + log_q_kl_bin + sub_dm_cond > log_single_double_precision_threshold) {
                ij_bin_double_first++;
                ij_bin_single_first++;
            } else if (log_q_ij_bin + log_q_kl_bin + sub_dm_cond > log_cutoff) {
                ij_bin_single_first++;
            } else {
                break;
            }
        }

        const int bas_ij0 = bins_locs_ij[0];
        const int bas_ij1 = bins_locs_ij[ij_bin_double_first];
        const int bas_ij2 = bins_locs_ij[ij_bin_single_first];
        const int ntasks_ij_double = bas_ij1 - bas_ij0;
        const int ntasks_ij_single = bas_ij2 - bas_ij1;

        if (ntasks_ij_double > 0) {
            offsets.ntasks_ij = ntasks_ij_double;
            offsets.ntasks_kl = ntasks_kl;
            offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
            offsets.bas_kl = bas_pairs_locs[cp_kl_id] + bas_kl0;
            offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * envs.nprim_ij;
            offsets.primitive_kl = primitive_pairs_locs[cp_kl_id] + bas_kl0 * envs.nprim_kl;

            JKMatrixMixedPrecision<double> jk_test;
            jk_test.n_dm = n_dm;
            jk_test.nao = nao;
            jk_test.dm = dm;
            jk_test.vj = vj;
            jk_test.vk = vk;

            n_tasks_count_double += offsets.ntasks_ij * offsets.ntasks_kl;

            const int err = GINTrun_tasks_jk<double>(&jk, &jk_test, &offsets, &envs);
            if (err != 0) {
                return err;
            }
        }

        if (ntasks_ij_single > 0) {
            offsets.ntasks_ij = ntasks_ij_single;
            offsets.ntasks_kl = ntasks_kl;
            offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij1;
            offsets.bas_kl = bas_pairs_locs[cp_kl_id] + bas_kl0;
            offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij1 * envs.nprim_ij;
            offsets.primitive_kl = primitive_pairs_locs[cp_kl_id] + bas_kl0 * envs.nprim_kl;

            JKMatrixMixedPrecision<float> jk_test;
            jk_test.n_dm = n_dm;
            jk_test.nao = nao;
            jk_test.dm = dm_single_precision;
            jk_test.vj = vj_single_precision;
            jk_test.vk = vk_single_precision;

            n_tasks_count_single += offsets.ntasks_ij * offsets.ntasks_kl;

            const int err = GINTrun_tasks_jk<float>(&jk, &jk_test, &offsets, &envs);
            if (err != 0) {
                return err;
            }
        }

    }

    if (envs.nf > NFffff) {
        FREE(envs.idx);
    }
    return 0;
}
}
