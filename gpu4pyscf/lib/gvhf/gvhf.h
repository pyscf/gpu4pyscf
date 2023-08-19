
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

#pragma once

#ifndef GPU4PYSCF_RYS_ROOTS_CUH
#define GPU4PYSCF_RYS_ROOTS_CUH

//extern __constant__ GINTEnvVars c_envs;
extern __constant__ BasisProdCache c_bpcache;
extern __constant__ int16_t c_idx4c[NFffff*3];
extern __constant__ int c_idx[TOT_NF*3];
extern __constant__ int c_l_locs[GPU_LMAX+2];

__device__ void GINTrys_root2(double x, double *rw);
__device__ void GINTrys_root3(double x, double *rw);
__device__ void GINTrys_root4(double x, double *rw);
__device__ void GINTrys_root5(double x, double *rw);

extern __constant__ BasisProdOffsets c_offsets[MAX_STREAMS];
extern __constant__ GINTEnvVars c_envs[MAX_STREAMS];
extern __constant__ JKMatrix c_jk[MAX_STREAMS];
#endif //GPU4PYSCF_RYS_ROOTS_CUH