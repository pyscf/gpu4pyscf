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
#include <cuda_runtime.h>

#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"
#include "cint2e.cuh"
#include "g2e.cu"
#include "g3c2e.cu"
#include "g3c2e_ip1.cu"
#include "g3c2e_ip2.cu"
#include "g3c2e_ipip1.cu"
#include "g3c2e_ip1ip2.cu"
#include "g3c2e_ipvip1.cu"
#include "g3c2e_ipip2.cu"

__host__
static int GINTfill_int3c2e_ip1_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream,)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    
    switch (envs->nrys_roots) {
        case 1: GINTfill_int3c2e_ip1_kernel1000<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 2: GINTfill_int3c2e_ip1_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 3: GINTfill_int3c2e_ip1_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 4: GINTfill_int3c2e_ip1_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 5: GINTfill_int3c2e_ip1_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 6: GINTfill_int3c2e_ip1_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 7: GINTfill_int3c2e_ip1_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 8: GINTfill_int3c2e_ip1_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        default:
            fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int2e_ip1_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
static int GINTfill_int3c2e_ip2_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    switch (envs->nrys_roots) {
        case 1: GINTfill_int3c2e_ip2_kernel0010<<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 2: GINTfill_int3c2e_ip2_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 3: GINTfill_int3c2e_ip2_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 4: GINTfill_int3c2e_ip2_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 5: GINTfill_int3c2e_ip2_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 6: GINTfill_int3c2e_ip2_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 7: GINTfill_int3c2e_ip2_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        case 8: GINTfill_int3c2e_ip2_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
        default:
            fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ip2_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
static int GINTfill_int3c2e_ipip_tasks(ERITensor *eri, BasisProdOffsets *offsets, GINTEnvVars *envs, int ip_type, cudaStream_t stream)
{
    int nrys_roots = envs->nrys_roots;
    int ntasks_ij = offsets->ntasks_ij;
    int ntasks_kl = offsets->ntasks_kl;
    assert(ntasks_kl < 65536*THREADSY);
    dim3 threads(THREADSX, THREADSY);
    dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ntasks_kl+THREADSY-1)/THREADSY);
    switch (envs->nrys_roots) {
        case 2:
            switch (ip_type){
                case 200: GINTfill_int3c2e_ipip1_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 101: GINTfill_int3c2e_ip1ip2_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 110: GINTfill_int3c2e_ipvip1_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 002: GINTfill_int3c2e_ipip2_kernel<2, GSIZE2_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            }
            break;
        case 3: 
            switch (ip_type){
                case 200: GINTfill_int3c2e_ipip1_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 101: GINTfill_int3c2e_ip1ip2_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 110: GINTfill_int3c2e_ipvip1_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 002: GINTfill_int3c2e_ipip2_kernel<3, GSIZE3_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            }
            break;
        case 4: 
            switch (ip_type){
                case 200: GINTfill_int3c2e_ipip1_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 101: GINTfill_int3c2e_ip1ip2_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 110: GINTfill_int3c2e_ipvip1_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 002: GINTfill_int3c2e_ipip2_kernel<4, GSIZE4_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            }
            break;
        case 5: 
            switch (ip_type){
                case 200: GINTfill_int3c2e_ipip1_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 101: GINTfill_int3c2e_ip1ip2_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 110: GINTfill_int3c2e_ipvip1_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 002: GINTfill_int3c2e_ipip2_kernel<5, GSIZE5_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            }
            break;
        case 6: 
            switch (ip_type){
                case 200: GINTfill_int3c2e_ipip1_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 101: GINTfill_int3c2e_ip1ip2_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 110: GINTfill_int3c2e_ipvip1_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 002: GINTfill_int3c2e_ipip2_kernel<6, GSIZE6_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            }
            break;
        case 7:
            switch (ip_type){
                case 200: GINTfill_int3c2e_ipip1_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 101: GINTfill_int3c2e_ip1ip2_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 110: GINTfill_int3c2e_ipvip1_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 002: GINTfill_int3c2e_ipip2_kernel<7, GSIZE7_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            }
            break;
        case 8: 
            switch (ip_type){
                case 200: GINTfill_int3c2e_ipip1_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 101: GINTfill_int3c2e_ip1ip2_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 110: GINTfill_int3c2e_ipvip1_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
                case 002: GINTfill_int3c2e_ipip2_kernel<8, GSIZE8_INT3C> <<<blocks, threads, 0, stream>>>(*envs, *eri, *offsets); break;
            }
            break;
        default:
            fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        size_t free, total, GB;
        GB = 1024 * 1024 * 1024;
        cudaMemGetInfo(&free, &total);
        fprintf(stderr, "-------------------- error info ------------------------------");
        fprintf(stderr, "CUDA Error of GINTfill_int3c2e_ipip_kernel: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "IP type: %d\n", ip_type);
        fprintf(stderr, "Angular momentum: (%d, %d, %d, %d)\n", envs->i_l, envs->j_l, envs->k_l, envs->l_l);
        fprintf(stderr, "%d GB free memory, %d GB total memory\n", free/GB, total/GB);
        fprintf(stderr, "----------------- end info -----------------------------------");
        return 1;
    }
    return 0;
}


extern "C" {


int GINTfill_int3c2e_ip(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id, int ip_type, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    int ng[4] = {0,0,0,0};
    if(ip_type == 1){
        ng[0] = 1;
    }
    else if(ip_type == 2){
        ng[2] = 1;
    }
    else{
         fprintf(stderr, "ip type unsupported\n");
    }
    
    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > 8) {
        return 2;
    }
    
    // TODO: improve the efficiency by unrolling
    if (envs.nrys_roots > 1) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        GINTg2e_index_xyz(idx4c, &envs);
        checkCudaErrors(cudaMemcpyToSymbol(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3));
        free(idx4c);
    }
    
    int kl_bin, ij_bin1;
    
    //checkCudaErrors(cudaMemcpyToSymbol(envs, &envs, sizeof(GINTEnvVars)));
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

        int err = -1;
        if(ip_type == 1){err = GINTfill_int3c2e_ip1_tasks(&eritensor, &offsets, &envs);}
        else 
        if(ip_type == 2){err = GINTfill_int3c2e_ip2_tasks(&eritensor, &offsets, &envs);}
        
        if (err != 0) {
            return err;
        }
    }
    
    return 0;
}

int GINTfill_int3c2e_general(cudaStream_t stream, BasisProdCache *bpcache, double *eri, int nao,
                   int *strides, int *ao_offsets,
                   int *bins_locs_ij, int *bins_locs_kl, int nbins,
                   int cp_ij_id, int cp_kl_id, int ip_type, double omega)
{
    ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    ContractionProdType *cp_kl = bpcache->cptype + cp_kl_id;
    GINTEnvVars envs;
    
    int ng[4] = {0,0,0,0};
    switch (ip_type){
        // 0-order
        case 000: break;
        // 1-order
        case 100: ng[0] = 1; break;
        case 001: ng[2] = 1; break;
        // 2-order
        case 200: ng[0] = 2; break;
        case 110: ng[0] = 1; ng[1] = 1; break;
        case 101: ng[0] = 1; ng[2] = 1; break;
        case 002: ng[2] = 2; break;
        // high-order not defined
        default: fprintf(stderr, "ip type unsupported\n");
    }
    
    GINTinit_EnvVars(&envs, cp_ij, cp_kl, ng);
    envs.omega = omega;
    if (envs.nrys_roots > 8) {
        return 2;
    }
    
    // TODO: improve the efficiency by unrolling
    if (envs.nrys_roots > 1) {
        int16_t *idx4c = (int16_t *)malloc(sizeof(int16_t) * envs.nf * 3);
        GINTg2e_index_xyz(idx4c, &envs);
        checkCudaErrors(cudaMemcpyToSymbol(c_idx4c, idx4c, sizeof(int16_t)*envs.nf*3));
        free(idx4c);
    }
    
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

        int err = -1;
        if (ng[0] + ng[1] + ng[2] == 2){
            err = GINTfill_int3c2e_ipip_tasks(&eritensor, &offsets, &envs, ip_type, stream);
        }
        else if (ng[0] + ng[1] + ng[2] == 1){
            if(ng[0] == 1){err = GINTfill_int3c2e_ip1_tasks(&eritensor, &offsets, &envs, stream);}
            if(ng[0] == 0){err = GINTfill_int3c2e_ip2_tasks(&eritensor, &offsets, &envs, stream);}
        }
        else if (ng[0] + ng[1] + ng[2] == 0){
            return -1;
            //err = GINTfill_int3c2e_tasks(&eritensor, &offsets, &envs);
        }
        
        if (err != 0) {
            return err;
        }
    }
    
    return 0;
}

}
