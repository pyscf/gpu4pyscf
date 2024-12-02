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

#pragma once

#include "cint2e.cuh"

template <int NROOTS> __device__
void GINTgout2e(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ g)
{
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;
    
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;

    const int nfi = envs.nfi;
    const int nfj = envs.nfj;
    const int nfk = envs.nfk;

    for (int ik = 0, i = 0; ik < nfk; ik++){
    for (int ij = 0; ij < nfj; ij++){
    for (int ii = 0; ii < nfi; ii++, i++){
        const int loc_k = c_l_locs[lk] + ik;
        const int loc_j = c_l_locs[lj] + ij;
        const int loc_i = c_l_locs[li] + ii;

        int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
        int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
        int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

        double s = gout[i];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n) {
            s += g[ix+n] * g[iy+n] * g[iz+n];
        }
        gout[i] = s;
    }}}
}
