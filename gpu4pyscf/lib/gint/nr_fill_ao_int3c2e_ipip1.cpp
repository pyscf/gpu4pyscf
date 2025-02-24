/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
#include "sycl_device.hpp"

#include "gint.h"
#include "config.h"
#include "sycl_alloc.hpp"
#include "g2e.h"
// #include "cint2e.hpp"

#include "rys_roots.cpp"
#include "g2e.cpp"
#include "gout3c2e.cpp"
#include "g3c2e_ipip1.cpp"


static int GINTfill_int3c2e_ipip1_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, sycl::queue& stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    sycl::range<2> threads(THREADSY, THREADSX);
    sycl::range<2> blocks((ntasks_kl+THREADSY-1)/THREADSY, (ntasks_ij+THREADSX-1)/THREADSX);
    switch (envs->nrys_roots) {
        case 2: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<2, GSIZE2_INT3C> (*envs, *eri, *offsets, item);}); break;
        case 3: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<3, GSIZE3_INT3C> (*envs, *eri, *offsets, item);}); break;
        case 4: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<4, GSIZE4_INT3C> (*envs, *eri, *offsets, item);}); break;
        case 5: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<5, GSIZE5_INT3C> (*envs, *eri, *offsets, item);}); break;
        case 6: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<6, GSIZE6_INT3C> (*envs, *eri, *offsets, item);}); break;
        case 7: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<7, GSIZE7_INT3C> (*envs, *eri, *offsets, item);}); break;
        case 8: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<8, GSIZE8_INT3C> (*envs, *eri, *offsets, item);}); break;
        case 9: stream.parallel_for(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { GINTfill_int3c2e_ipip1_kernel<9, GSIZE9_INT3C> (*envs, *eri, *offsets, item);}); break;
        default:
            fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ipip1_kernel: %s\n", cudaGetErrorString(err));
    //     return 1;
    // }
    return 0;
}

extern "C" {
int GINTfill_int3c2e_ipip1(sycl::queue& stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;

    int ng[4] = {2,0,0,0};

    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > 9) {
        return 2;
    }

    // TODO: improve the efficiency by unrolling
    if (envs.nrys_roots > 1) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        GINTg2e_index_xyz(idx4c, &envs);
        stream.memcpy(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3);
        free(idx4c);
    }

    int kl_bin, ij_bin1;

    //checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
    // move bpcache to constant memory
    stream.memcpy(c_bpcache, bpcache, sizeof(BasisProdCache));

    ERITensor eritensor;
    eritensor.stride_j = strides[1];
    eritensor.stride_k = strides[2];
    eritensor.stride_l = strides[3];
    eritensor.ao_offsets_i = ao_offsets[0];
    eritensor.ao_offsets_j = ao_offsets[1];
    eritensor.ao_offsets_k = ao_offsets[2];
    eritensor.ao_offsets_l = ao_offsets[3];
    eritensor.nao = nao;
    eritensor.data = eri;
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

        int err = GINTfill_int3c2e_ipip1_tasks(&eritensor, &offsets, &envs, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

}
