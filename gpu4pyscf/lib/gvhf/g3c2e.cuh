/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "gint/g2e.h"
#include "gint/cint2e.cuh"
#include "gvhf.h"

// jaux = numpy.einsum('ijk,ji->k', j3c, dm)
template <int NROOTS> __device__
static void GINTkernel_int3c2e_getj_pass1(GINTEnvVars envs, JKMatrix jk, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    int nao = jk.nao;
    int i, j, k;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ dm = jk.dm;

    int i_l = envs.i_l;
    int j_l = envs.j_l;
    int k_l = envs.k_l;
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    for (k = k0; k < k1; ++k) {
        int kp = k - k0;
        double rhoj_tmp = 0.0;
        for (j = j0; j < j1; ++j) {
            int jp = j - j0;
            for (i = i0; i < i1; ++i) {
                int ip = i - i0;

                int loc_k = c_l_locs[k_l] + kp;
                int loc_j = c_l_locs[j_l] + jp;
                int loc_i = c_l_locs[i_l] + ip;

                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + envs.g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + envs.g_size * 2;
                double s = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    s += g[ix+ir] * g[iy+ir] * g[iz+ir];
                }
                int off_dm = i + nao*j;
                rhoj_tmp += dm[off_dm] * s;
            }
        }
        atomicAdd(rhoj+k, rhoj_tmp);
    }
}

// vj = numpy.einsum('ijk,k->ij', j3c, rho)
template <int NROOTS> __device__
static void GINTkernel_int3c2e_getj_pass2(GINTEnvVars envs, JKMatrix jk, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    int nao = jk.nao;
    int i, j, k;
    double* __restrict__ vj = jk.vj;

    int i_l = envs.i_l;
    int j_l = envs.j_l;
    int k_l = envs.k_l;
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    double rhoj[GPU_CART_MAX];
    for (k = 0; k < k1-k0; k++){
        rhoj[k] = jk.rhoj[k0+k];
    }

    for (j = j0; j < j1; ++j) {
        int jp = j - j0;
        for (i = i0; i < i1; ++i) {
            int ip = i - i0;
            double vj_tmp = 0.0;
            for (k = k0; k < k1; ++k){
                int kp = k - k0;

                int loc_k = c_l_locs[k_l] + kp;
                int loc_j = c_l_locs[j_l] + jp;
                int loc_i = c_l_locs[i_l] + ip;

                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + envs.g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + envs.g_size * 2;
                double s = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    s += g[ix+ir] * g[iy+ir] * g[iz+ir];
                }
                vj_tmp += rhoj[k-k0] * s;
            }
            atomicAdd(vj+j+nao*i, vj_tmp);
        }
    }
}


