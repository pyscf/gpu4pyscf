/*
 * gpu4pyscf is a plugin to use Nvidia/AMD/Intel GPU in PySCF package
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

#include "gint/gint.h"
#include "gint/config.h"
#include "gint/sycl_alloc.hpp"
#include "gint/g2e.h"
#include "gint/cint2e.hpp"

#include "contract_jk.cpp"
#include "gint/rys_roots.cpp"

#include "g2e.cpp"
#include "g2e_root2.cpp"
#include "g2e_root3.cpp"

static int GINTrun_tasks_jk(JKMatrix *jk, BasisProdOffsets *offsets, GINTEnvVars *envs)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    int type_ijkl;
    sycl::range<2> threads(THREADSY, THREADSX);
    sycl::range<2> blocks((ntasks_kl+THREADSY-1)/THREADSY, (ntasks_ij+THREADSX-1)/THREADSX);
    switch (nrys_roots) {
    case 1:
        if (envs->nf == 1) {
            sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel0000(*envs, *jk, *offsets, item); });
	} else {
	    sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel1000(*envs, *jk, *offsets, item); });
        }
        break;
    case 2:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (1<<6)|(0<<4)|(1<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel1010(*envs, *jk, *offsets, item); }); break;
        case (1<<6)|(0<<4)|(1<<2)|1: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel1011(*envs, *jk, *offsets, item); }); break;
        case (1<<6)|(1<<4)|(0<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel1100(*envs, *jk, *offsets, item); }); break;
        case (1<<6)|(1<<4)|(1<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel1110(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(0<<4)|(0<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2000(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(0<<4)|(1<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2010(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(1<<4)|(0<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2100(*envs, *jk, *offsets, item); }); break;
        case (3<<6)|(0<<4)|(0<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel3000(*envs, *jk, *offsets, item); }); break;
        default:
            sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<2, GSIZE2>(*envs, *jk, *offsets, item); }); break;
        }
        break;
    case 3:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (1<<6)|(1<<4)|(1<<2)|1: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel1111(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(0<<4)|(1<<2)|1: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2011(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(0<<4)|(2<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2020(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(0<<4)|(2<<2)|1: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2021(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(1<<4)|(1<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2110(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(1<<4)|(1<<2)|1: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2111(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(1<<4)|(2<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2120(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(2<<4)|(0<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2200(*envs, *jk, *offsets, item); }); break;
        case (2<<6)|(2<<4)|(1<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel2210(*envs, *jk, *offsets, item); }); break;
        case (3<<6)|(0<<4)|(1<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel3010(*envs, *jk, *offsets, item); }); break;
        case (3<<6)|(0<<4)|(1<<2)|1: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel3011(*envs, *jk, *offsets, item); }); break;
        case (3<<6)|(0<<4)|(2<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel3020(*envs, *jk, *offsets, item); }); break;
        case (3<<6)|(1<<4)|(0<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel3100(*envs, *jk, *offsets, item); }); break;
        case (3<<6)|(1<<4)|(1<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel3110(*envs, *jk, *offsets, item); }); break;
        case (3<<6)|(2<<4)|(0<<2)|0: sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel3200(*envs, *jk, *offsets, item); }); break;
        default:
            sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<3, GSIZE3>(*envs, *jk, *offsets, item); }); break;
        }
        break;
    case 4:
        sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<4, GSIZE4>(*envs, *jk, *offsets, item); }); break;
    case 5:
        sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<5, GSIZE5>(*envs, *jk, *offsets, item); }); break;
    case 6:
        sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<6, GSIZE6>(*envs, *jk, *offsets, item); }); break;
    case 7:
        sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<7, GSIZE7>(*envs, *jk, *offsets, item); }); break;
    case 8:
        sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<8, GSIZE8>(*envs, *jk, *offsets, item); }); break;
    case 9:
        sycl_get_queue()->parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTint2e_jk_kernel<9, GSIZE9>(*envs, *jk, *offsets, item); }); break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA Error of GINTint2e_jk_kernel: %s\n", cudaGetErrorString(err));
    //     return 1;
    // }
    return 0;
}

extern "C" {
int GINTbuild_jk(sycl::queue stream, BasisProdCache *bpcache,
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
    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > POLYFIT_ORDER) {
        fprintf(stderr, "build_jk: unsupported rys order %d\n", envs.nrys_roots);
        return 2;
    }

    if (envs.nrys_roots > 2) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        int *idx_ij = (int *)malloc(sizeof(int) * envs.nfi * envs.nfj * 3);
        int *idx_kl = (int *)malloc(sizeof(int) * envs.nfk * envs.nfl * 3);
        GINTinit_2c_gidx(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
        GINTinit_2c_gidx(idx_kl, cp_kl->l_bra, cp_kl->l_ket);
        GINTinit_4c_idx(idx4c, idx_ij, idx_kl, &envs);

        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     fprintf(stderr, "CUDA Error of GINTbuild_int2e_kernel: %s\n", cudaGetErrorString(err));
        //     return 1;
        // }

        if (envs.nf > NFffff) {
            sycl::queue q = stream;
            DEVICE_INIT(int16_t, d_idx4c, idx4c, envs.nf * 3);
            envs.idx = d_idx4c;
        } else {
	    sycl_get_queue()->memcpy(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3).wait();
            //checkCudaErrors(cudaMemcpyToSymbol(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3));
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
    //checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
    // move bpcache to constant memory
    sycl_get_queue()->memcpy(c_bpcache, bpcache, sizeof(BasisProdCache)).wait();
    //checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    JKMatrix jk;
    jk.n_dm = n_dm;
    jk.nao = nao;
    jk.dm = dm;
    jk.vj = vj;
    jk.vk = vk;
    jk.dm_sh = dm_sh;
    jk.nshls = nshls;
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

        // ij_bin1 is the index of first bin out of cutoff
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

        int err = GINTrun_tasks_jk(&jk, &offsets, &envs);
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
