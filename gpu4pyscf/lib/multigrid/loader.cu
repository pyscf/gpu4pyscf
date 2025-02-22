/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "multigrid.cuh"

__device__ static
void init_orth_data(double *pool, int *grid_start,
                    MGridEnvVars envs, MGridBounds bounds, double *ri, double *rj,
                    double ai, double aj, int l1, int warp_id)
{
    int ngrid_span = bounds.ngrid_radius * 2;
    int nl_gridx = ngrid_span * l1;
    int xs_size = nl_gridx * WARP_SIZE;
    int z = warp_id % 3;
    int w = warp_id / 3;
    double aij = ai + aj;
    double aj_aij = aj / aij;
    if (w < 2) {
        int nx_per_cell = bounds.mesh[z];
        double *exp_cache = pool + z * xs_size;
        double xjxi = rj[z] - ri[z];
        double xij = xjxi * aj_aij + ri[z];
        double dx = envs.lattice_params[z] / nx_per_cell;
        int xij_latt = static_cast<int>(floor(xij / dx));
        if (w == 0) {
            int x0_latt = xij_latt + 1 - bounds.ngrid_radius;
            grid_start[z*WARP_SIZE] = x0_latt;
        }
        double base_x = dx * xij_latt;
        double x0xij = base_x - xij;
        double _x0x0 = -aij * x0xij * x0xij;
        double _dxdx = -aij * dx * dx;
        double _x0dx = -2 * aij * x0xij * dx;
        double exp_2dxdx = exp(2 * _dxdx);
        double exp_x0x0 = exp(_x0x0);
        int istart = bounds.ngrid_radius - 1;
        if (_x0x0 < -40) { // 2e-18
            exp_x0x0 = 0.;
            exp_2dxdx = 0.;
            _x0dx = 0.; // exp(_x0dx) may singular when aij is large
        }
        if (w == 0) {
            double exp_x0dx = exp(_dxdx - _x0dx);
            for (int i = istart; i >= 0; --i) {
                exp_cache[i*WARP_SIZE] = exp_x0x0;
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
            }
        } else {
            double exp_x0dx = exp(_dxdx + _x0dx);
            for (int i = istart+1; i < ngrid_span; ++i) {
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
                exp_cache[i*WARP_SIZE] = exp_x0x0;
            }
        }
    }
    __syncthreads();

    for (int z = 0; z < 3; ++z) {
        int nx_per_cell = bounds.mesh[z];
        int x0_latt = grid_start[z*WARP_SIZE];
        double dx = envs.lattice_params[z] / nx_per_cell;
        double x0xi = x0_latt * dx - ri[z];
        double *xs_exp = pool + z * xs_size;

        for (int i = warp_id; i < ngrid_span; i += WARPS) {
            double gridx = x0xi + i * dx;
            double s1 = xs_exp[i*WARP_SIZE];
            for (int m = 1; m < l1; m++) {
                s1 *= gridx;
                xs_exp[(m*ngrid_span+i)*WARP_SIZE] = s1;
            }
        }
        __syncthreads();

        // add up contributions from all images to the reference image
        if (ngrid_span >= nx_per_cell) {
            for (int n = warp_id; n < nx_per_cell*l1; n += WARPS) {
                int i = n % nx_per_cell;
                int m = n / nx_per_cell;
                double s1 = 0.;
                double *xe = xs_exp + m*ngrid_span*WARP_SIZE;
                for (; i < ngrid_span; i += nx_per_cell) {
                    s1 += xe[i*WARP_SIZE];
                }
                i = n % nx_per_cell;
                xs_exp[(m*ngrid_span + i)*WARP_SIZE] = s1;
            }
        }
        if (warp_id == 0) {
            x0_latt = (x0_latt % nx_per_cell + nx_per_cell) % nx_per_cell;
            grid_start[z*WARP_SIZE] = x0_latt;
        }
    }
    __syncthreads();
}
