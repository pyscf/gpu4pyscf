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
    int nf = envs.nf;
    int16_t *idx = c_idx4c;
    if (nf > NFffff) {
        idx = envs.idx;
    }
    int16_t *idy = idx + nf;
    int16_t *idz = idx + nf * 2;
    double s;
    int i, n, ix, iy, iz;
    
    for (i = 0; i < nf; i++) {
        ix = idx[i];
        iy = idy[i];
        iz = idz[i];
        s = gout[i];
#pragma unroll
        for (n = 0; n < NROOTS; ++n) {
            s += g[ix+n] * g[iy+n] * g[iz+n];
        }
        gout[i] = s;
    }
}
