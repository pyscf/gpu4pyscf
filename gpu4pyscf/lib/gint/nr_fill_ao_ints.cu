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

#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"
#include "g2e.cu"
#include "cint2e.cuh"
#include "rys_roots.cu"
#include "gout2e.cuh"

#include "fill_ints.cu"
#include "g2e_root1.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"
#include "g2e_root_n.cu"

__host__
static int GINTfill_int2e_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    int type_ijkl;
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    switch (nrys_roots) {
    case 1:
        type_ijkl = (envs->i_l << 3) | (envs->j_l << 2) | (envs->k_l << 1) | envs->l_l;
        switch (type_ijkl) {
        case 0b0000: GINTfill_int2e_kernel0000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 0b0010: GINTfill_int2e_kernel0010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 0b1000: GINTfill_int2e_kernel1000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        default:
            //GINTfill_int2e_kernel<1, GOUTSIZE1> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (0<<6)|(0<<4)|(1<<2)|1: GINTfill_int2e_kernel0011<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (0<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel0020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (0<<6)|(0<<4)|(2<<2)|1: GINTfill_int2e_kernel0021<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (0<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel0030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel1010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(0<<4)|(1<<2)|1: GINTfill_int2e_kernel1011<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel1020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel1100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel1110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(0<<4)|(0<<2)|0: GINTfill_int2e_kernel2000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel2010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel2100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (3<<6)|(0<<4)|(0<<2)|0: GINTfill_int2e_kernel3000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        default:
            GINTfill_int2e_kernel<2, GOUTSIZE2> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        }
        break;
    case 3:
        type_ijkl = (envs->i_l << 6) | (envs->j_l << 4) | (envs->k_l << 2) | envs->l_l;
        switch (type_ijkl) {
        case (0<<6)|(0<<4)|(2<<2)|2: GINTfill_int2e_kernel0022<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (0<<6)|(0<<4)|(3<<2)|1: GINTfill_int2e_kernel0031<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (0<<6)|(0<<4)|(3<<2)|2: GINTfill_int2e_kernel0032<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(0<<4)|(2<<2)|1: GINTfill_int2e_kernel1021<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(0<<4)|(2<<2)|2: GINTfill_int2e_kernel1022<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel1030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(0<<4)|(3<<2)|1: GINTfill_int2e_kernel1031<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(1<<4)|(1<<2)|1: GINTfill_int2e_kernel1111<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(1<<4)|(2<<2)|0: GINTfill_int2e_kernel1120<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(1<<4)|(2<<2)|1: GINTfill_int2e_kernel1121<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (1<<6)|(1<<4)|(3<<2)|0: GINTfill_int2e_kernel1130<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(0<<4)|(1<<2)|1: GINTfill_int2e_kernel2011<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel2020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(0<<4)|(2<<2)|1: GINTfill_int2e_kernel2021<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(0<<4)|(3<<2)|0: GINTfill_int2e_kernel2030<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel2110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(1<<4)|(1<<2)|1: GINTfill_int2e_kernel2111<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(1<<4)|(2<<2)|0: GINTfill_int2e_kernel2120<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(2<<4)|(0<<2)|0: GINTfill_int2e_kernel2200<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (2<<6)|(2<<4)|(1<<2)|0: GINTfill_int2e_kernel2210<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|0: GINTfill_int2e_kernel3010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (3<<6)|(0<<4)|(1<<2)|1: GINTfill_int2e_kernel3011<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (3<<6)|(0<<4)|(2<<2)|0: GINTfill_int2e_kernel3020<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (3<<6)|(1<<4)|(0<<2)|0: GINTfill_int2e_kernel3100<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (3<<6)|(1<<4)|(1<<2)|0: GINTfill_int2e_kernel3110<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case (3<<6)|(2<<4)|(0<<2)|0: GINTfill_int2e_kernel3200<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        default:
            GINTfill_int2e_kernel<3, GOUTSIZE3> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        }
        break;
    case 4: GINTfill_int2e_kernel<4, GOUTSIZE4> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    case 5: GINTfill_int2e_kernel<5, GOUTSIZE5> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    case 6: GINTfill_int2e_kernel<6, GOUTSIZE6> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    case 7: GINTfill_int2e_kernel<7, GOUTSIZE7> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    case 8: GINTfill_int2e_kernel<8, GOUTSIZE8> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int2e_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTfill_int2e(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, 
                   double *bins_floor_ij, double *bins_floor_kl,
                   int nbins_ij, int nbins_kl,
                   int cp_ij_id, int cp_kl_id, double log_cutoff, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    
    int ng[4] = {0,0,0,0};
    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > POLYFIT_ORDER) {
        fprintf(stderr, "GINTfill_int2e: unsupported rys order %d\n", envs.nrys_roots);
        return 2;
    }
    
    if (envs.nrys_roots > 2) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        int *idx_ij = (int *)malloc(sizeof(int) * envs.nfi * envs.nfj * 3);
        int *idx_kl = (int *)malloc(sizeof(int) * envs.nfk * envs.nfl * 3);
        GINTinit_2c_gidx(idx_ij, cp_ij->l_bra, cp_ij->l_ket);
        GINTinit_2c_gidx(idx_kl, cp_kl->l_bra, cp_kl->l_ket);
        GINTinit_4c_idx(idx4c, idx_ij, idx_kl, &envs);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error of GINTfill_int2e_kernel: %s\n", cudaGetErrorString(err));
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
    int kl_bin, ij_bin1;
    //checkCudaErrors(cudaMemcpyToSymbol(c_envs, &envs, sizeof(GINTEnvVars)));
    // move bpcache to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
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
            if (log_q_ij_bin + log_q_kl_bin < log_cutoff){
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
        
        int err = GINTfill_int2e_tasks(&eritensor, &offsets, &envs, stream);
        if (err != 0) {
            return err;
        }
    }
    
    if (envs.nrys_roots > 2) {
        if (envs.nf > NFffff) {
            FREE(envs.idx);
        }
    }
    return 0;
}
}
