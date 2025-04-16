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

#include "cint2e.cuh"

template <int NROOTS> __device__
void GINTgout2e(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ g)
{
    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;
    
    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int dl = envs.stride_l;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;
    const int ll = envs.l_l;

    const int nfi = envs.nfi;
    const int nfj = envs.nfj;
    const int nfk = envs.nfk;
    const int nfl = envs.nfl;

    for (int il = 0, i = 0; il < nfl; il++){
    for (int ik = 0; ik < nfk; ik++){
    for (int ij = 0; ij < nfj; ij++){
    for (int ii = 0; ii < nfi; ii++, i++){
        const int loc_l = c_l_locs[ll] + il;
        const int loc_k = c_l_locs[lk] + ik;
        const int loc_j = c_l_locs[lj] + ij;
        const int loc_i = c_l_locs[li] + ii;

        const int ix = dl * idx[loc_l] + dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
        const int iy = dl * idy[loc_l] + dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
        const int iz = dl * idz[loc_l] + dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

        double s = gout[i];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n) {
            s += g[ix+n] * g[iy+n] * g[iz+n];
        }
        gout[i] = s;
    }}}}
}
