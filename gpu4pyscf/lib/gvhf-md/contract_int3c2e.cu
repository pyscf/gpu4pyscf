/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/boys.cu"
#include "gvhf-md/md_j.cuh"

#define RT2_MAX 9
#define IJ_SIZE 11
#define THREADS 256
#define L_AUX_MAX 6

__device__
inline void iter_Rt_n(double *out, double *Rt, double rx, double ry, double rz, int l,
                      int nst_per_block, int gout_id, int gout_stride)
{
    int offsets = l*(l+1)*(l+2)*(l+3)/24;
    uint16_t *p1 = c_Rt_idx + offsets - l;
    double *pout = out + nst_per_block;
    for (int v = gout_id; v < l; v += gout_stride) {
        pout[v*nst_per_block] = rz * Rt[v*nst_per_block] + v * Rt[p1[v]*nst_per_block];
    }
    pout += l * nst_per_block;
    p1 += l;
    int8_t *tuv_fac = c_Rt_tuv_fac + offsets;

    int n2 = l * (l+1) / 2;
    for (int i = gout_id; i < n2; i += gout_stride) {
        pout[i*nst_per_block] = ry * Rt[i*nst_per_block] + tuv_fac[i] * Rt[p1[i]*nst_per_block];
    }
    pout += n2 * nst_per_block;
    p1 += n2;
    tuv_fac += n2;

    int n3 = n2 * (l+2) / 3;
    for (int i = gout_id; i < n3; i += gout_stride) {
        pout[i*nst_per_block] = rx * Rt[i*nst_per_block] + tuv_fac[i] * Rt[p1[i]*nst_per_block];
    }
}

/*
Et_base is generated using the recursion
def get_E_tensor(l):
    Et = np.zeros((l+1, l+1))
    pow_a = np.zeros((l+1, l+1), dtype=int)
    Et[0,0] = 1
    for i in range(1, l+1):
        Et[i,0] = Et[i-1,1]
        pow_a[i,0] = pow_a[i-1,1]
        for t in range(1, l+1):
            Et[i,t] = i/(2*t)*Et[i-1,t-1]
            pow_a[i,t] = pow_a[i-1,t-1] + 1

    for i, (ix, iy, iz) in enumerate((ix, iy, l-ix-iy) for ix in reversed(range(l+1)) for iy in reversed(range(l+1-ix))):
        for n, (t, u, v) in enumerate((t, u, v) for t in range(l+1) for u in range(l+1-t) for v in range(l+1-t-u)):
            if Et[ix,t] * Et[iy,u] * Et[iz,v] != 0:
                a_idx = pow_a[ix,t] + pow_a[iy,u] + pow_a[iz,v] - 1
                if a_idx >= 0:
                    print(f'out[{i}] += Rt[{n}] * aa[{a_idx}] * {Et[ix,t] * Et[iy,u] * Et[iz,v]};')
                else:
                    print(f'out[{i}] += Rt[{n}] * {Et[ix,t] * Et[iy,u] * Et[iz,v]};')
 */
template <int L> __device__ inline
void _dot_Et(double *out, double *Rt, double ai)
{
    double aa[L+1];
    aa[0] = 1 / ai;
#pragma unroll
    for (int n = 1; n < L; n++) {
        aa[n] = aa[n-1] * aa[0];
    }
    if constexpr (L == 0) {
        out[0] += Rt[0];
    } else if constexpr (L == 1) {
        out[0] += Rt[3] * aa[0] * 0.5;
        out[1] += Rt[2] * aa[0] * 0.5;
        out[2] += Rt[1] * aa[0] * 0.5;
    } else if constexpr (L == 2) {
        out[0] += Rt[0] * aa[0] * 0.5;
        out[0] += Rt[9] * aa[1] * 0.25;
        out[1] += Rt[8] * aa[1] * 0.25;
        out[2] += Rt[7] * aa[1] * 0.25;
        out[3] += Rt[0] * aa[0] * 0.5;
        out[3] += Rt[5] * aa[1] * 0.25;
        out[4] += Rt[4] * aa[1] * 0.25;
        out[5] += Rt[0] * aa[0] * 0.5;
        out[5] += Rt[2] * aa[1] * 0.25;
    } else if constexpr (L == 3) {
        out[0] += Rt[10] * aa[1] * 0.75;
        out[0] += Rt[19] * aa[2] * 0.125;
        out[1] += Rt[4] * aa[1] * 0.25;
        out[1] += Rt[18] * aa[2] * 0.125;
        out[2] += Rt[1] * aa[1] * 0.25;
        out[2] += Rt[17] * aa[2] * 0.125;
        out[3] += Rt[10] * aa[1] * 0.25;
        out[3] += Rt[15] * aa[2] * 0.125;
        out[4] += Rt[14] * aa[2] * 0.125;
        out[5] += Rt[10] * aa[1] * 0.25;
        out[5] += Rt[12] * aa[2] * 0.125;
        out[6] += Rt[4] * aa[1] * 0.75;
        out[6] += Rt[9] * aa[2] * 0.125;
        out[7] += Rt[1] * aa[1] * 0.25;
        out[7] += Rt[8] * aa[2] * 0.125;
        out[8] += Rt[4] * aa[1] * 0.25;
        out[8] += Rt[6] * aa[2] * 0.125;
        out[9] += Rt[1] * aa[1] * 0.75;
        out[9] += Rt[3] * aa[2] * 0.125;
    } else if constexpr (L == 4) {
        out[0] += Rt[0] * aa[1] * 0.75;
        out[0] += Rt[25] * aa[2] * 0.75;
        out[0] += Rt[34] * aa[3] * 0.0625;
        out[1] += Rt[19] * aa[2] * 0.375;
        out[1] += Rt[33] * aa[3] * 0.0625;
        out[2] += Rt[16] * aa[2] * 0.375;
        out[2] += Rt[32] * aa[3] * 0.0625;
        out[3] += Rt[0] * aa[1] * 0.25;
        out[3] += Rt[9] * aa[2] * 0.125;
        out[3] += Rt[25] * aa[2] * 0.125;
        out[3] += Rt[30] * aa[3] * 0.0625;
        out[4] += Rt[6] * aa[2] * 0.125;
        out[4] += Rt[29] * aa[3] * 0.0625;
        out[5] += Rt[0] * aa[1] * 0.25;
        out[5] += Rt[2] * aa[2] * 0.125;
        out[5] += Rt[25] * aa[2] * 0.125;
        out[5] += Rt[27] * aa[3] * 0.0625;
        out[6] += Rt[19] * aa[2] * 0.375;
        out[6] += Rt[24] * aa[3] * 0.0625;
        out[7] += Rt[16] * aa[2] * 0.125;
        out[7] += Rt[23] * aa[3] * 0.0625;
        out[8] += Rt[19] * aa[2] * 0.125;
        out[8] += Rt[21] * aa[3] * 0.0625;
        out[9] += Rt[16] * aa[2] * 0.375;
        out[9] += Rt[18] * aa[3] * 0.0625;
        out[10] += Rt[0] * aa[1] * 0.75;
        out[10] += Rt[9] * aa[2] * 0.75;
        out[10] += Rt[14] * aa[3] * 0.0625;
        out[11] += Rt[6] * aa[2] * 0.375;
        out[11] += Rt[13] * aa[3] * 0.0625;
        out[12] += Rt[0] * aa[1] * 0.25;
        out[12] += Rt[2] * aa[2] * 0.125;
        out[12] += Rt[9] * aa[2] * 0.125;
        out[12] += Rt[11] * aa[3] * 0.0625;
        out[13] += Rt[6] * aa[2] * 0.375;
        out[13] += Rt[8] * aa[3] * 0.0625;
        out[14] += Rt[0] * aa[1] * 0.75;
        out[14] += Rt[2] * aa[2] * 0.75;
        out[14] += Rt[4] * aa[3] * 0.0625;
    } else if constexpr (L == 5) {
        out[0] += Rt[21] * aa[2] * 1.875;
        out[0] += Rt[46] * aa[3] * 0.625;
        out[0] += Rt[55] * aa[4] * 0.03125;
        out[1] += Rt[6] * aa[2] * 0.375;
        out[1] += Rt[40] * aa[3] * 0.375;
        out[1] += Rt[54] * aa[4] * 0.03125;
        out[2] += Rt[1] * aa[2] * 0.375;
        out[2] += Rt[37] * aa[3] * 0.375;
        out[2] += Rt[53] * aa[4] * 0.03125;
        out[3] += Rt[21] * aa[2] * 0.375;
        out[3] += Rt[30] * aa[3] * 0.1875;
        out[3] += Rt[46] * aa[3] * 0.0625;
        out[3] += Rt[51] * aa[4] * 0.03125;
        out[4] += Rt[27] * aa[3] * 0.1875;
        out[4] += Rt[50] * aa[4] * 0.03125;
        out[5] += Rt[21] * aa[2] * 0.375;
        out[5] += Rt[23] * aa[3] * 0.1875;
        out[5] += Rt[46] * aa[3] * 0.0625;
        out[5] += Rt[48] * aa[4] * 0.03125;
        out[6] += Rt[6] * aa[2] * 0.375;
        out[6] += Rt[15] * aa[3] * 0.0625;
        out[6] += Rt[40] * aa[3] * 0.1875;
        out[6] += Rt[45] * aa[4] * 0.03125;
        out[7] += Rt[1] * aa[2] * 0.125;
        out[7] += Rt[12] * aa[3] * 0.0625;
        out[7] += Rt[37] * aa[3] * 0.0625;
        out[7] += Rt[44] * aa[4] * 0.03125;
        out[8] += Rt[6] * aa[2] * 0.125;
        out[8] += Rt[8] * aa[3] * 0.0625;
        out[8] += Rt[40] * aa[3] * 0.0625;
        out[8] += Rt[42] * aa[4] * 0.03125;
        out[9] += Rt[1] * aa[2] * 0.375;
        out[9] += Rt[3] * aa[3] * 0.0625;
        out[9] += Rt[37] * aa[3] * 0.1875;
        out[9] += Rt[39] * aa[4] * 0.03125;
        out[10] += Rt[21] * aa[2] * 0.375;
        out[10] += Rt[30] * aa[3] * 0.375;
        out[10] += Rt[35] * aa[4] * 0.03125;
        out[11] += Rt[27] * aa[3] * 0.1875;
        out[11] += Rt[34] * aa[4] * 0.03125;
        out[12] += Rt[21] * aa[2] * 0.125;
        out[12] += Rt[23] * aa[3] * 0.0625;
        out[12] += Rt[30] * aa[3] * 0.0625;
        out[12] += Rt[32] * aa[4] * 0.03125;
        out[13] += Rt[27] * aa[3] * 0.1875;
        out[13] += Rt[29] * aa[4] * 0.03125;
        out[14] += Rt[21] * aa[2] * 0.375;
        out[14] += Rt[23] * aa[3] * 0.375;
        out[14] += Rt[25] * aa[4] * 0.03125;
        out[15] += Rt[6] * aa[2] * 1.875;
        out[15] += Rt[15] * aa[3] * 0.625;
        out[15] += Rt[20] * aa[4] * 0.03125;
        out[16] += Rt[1] * aa[2] * 0.375;
        out[16] += Rt[12] * aa[3] * 0.375;
        out[16] += Rt[19] * aa[4] * 0.03125;
        out[17] += Rt[6] * aa[2] * 0.375;
        out[17] += Rt[8] * aa[3] * 0.1875;
        out[17] += Rt[15] * aa[3] * 0.0625;
        out[17] += Rt[17] * aa[4] * 0.03125;
        out[18] += Rt[1] * aa[2] * 0.375;
        out[18] += Rt[3] * aa[3] * 0.0625;
        out[18] += Rt[12] * aa[3] * 0.1875;
        out[18] += Rt[14] * aa[4] * 0.03125;
        out[19] += Rt[6] * aa[2] * 0.375;
        out[19] += Rt[8] * aa[3] * 0.375;
        out[19] += Rt[10] * aa[4] * 0.03125;
        out[20] += Rt[1] * aa[2] * 1.875;
        out[20] += Rt[3] * aa[3] * 0.625;
        out[20] += Rt[5] * aa[4] * 0.03125;
    } else if constexpr (L == 6) {
        out[0] += Rt[0] * aa[2] * 1.875;
        out[0] += Rt[49] * aa[3] * 2.8125;
        out[0] += Rt[74] * aa[4] * 0.46875;
        out[0] += Rt[83] * aa[5] * 0.015625;
        out[1] += Rt[34] * aa[3] * 0.9375;
        out[1] += Rt[68] * aa[4] * 0.3125;
        out[1] += Rt[82] * aa[5] * 0.015625;
        out[2] += Rt[29] * aa[3] * 0.9375;
        out[2] += Rt[65] * aa[4] * 0.3125;
        out[2] += Rt[81] * aa[5] * 0.015625;
        out[3] += Rt[0] * aa[2] * 0.375;
        out[3] += Rt[13] * aa[3] * 0.1875;
        out[3] += Rt[49] * aa[3] * 0.375;
        out[3] += Rt[58] * aa[4] * 0.1875;
        out[3] += Rt[74] * aa[4] * 0.03125;
        out[3] += Rt[79] * aa[5] * 0.015625;
        out[4] += Rt[8] * aa[3] * 0.1875;
        out[4] += Rt[55] * aa[4] * 0.1875;
        out[4] += Rt[78] * aa[5] * 0.015625;
        out[5] += Rt[0] * aa[2] * 0.375;
        out[5] += Rt[2] * aa[3] * 0.1875;
        out[5] += Rt[49] * aa[3] * 0.375;
        out[5] += Rt[51] * aa[4] * 0.1875;
        out[5] += Rt[74] * aa[4] * 0.03125;
        out[5] += Rt[76] * aa[5] * 0.015625;
        out[6] += Rt[34] * aa[3] * 0.5625;
        out[6] += Rt[43] * aa[4] * 0.09375;
        out[6] += Rt[68] * aa[4] * 0.09375;
        out[6] += Rt[73] * aa[5] * 0.015625;
        out[7] += Rt[29] * aa[3] * 0.1875;
        out[7] += Rt[40] * aa[4] * 0.09375;
        out[7] += Rt[65] * aa[4] * 0.03125;
        out[7] += Rt[72] * aa[5] * 0.015625;
        out[8] += Rt[34] * aa[3] * 0.1875;
        out[8] += Rt[36] * aa[4] * 0.09375;
        out[8] += Rt[68] * aa[4] * 0.03125;
        out[8] += Rt[70] * aa[5] * 0.015625;
        out[9] += Rt[29] * aa[3] * 0.5625;
        out[9] += Rt[31] * aa[4] * 0.09375;
        out[9] += Rt[65] * aa[4] * 0.09375;
        out[9] += Rt[67] * aa[5] * 0.015625;
        out[10] += Rt[0] * aa[2] * 0.375;
        out[10] += Rt[13] * aa[3] * 0.375;
        out[10] += Rt[22] * aa[4] * 0.03125;
        out[10] += Rt[49] * aa[3] * 0.1875;
        out[10] += Rt[58] * aa[4] * 0.1875;
        out[10] += Rt[63] * aa[5] * 0.015625;
        out[11] += Rt[8] * aa[3] * 0.1875;
        out[11] += Rt[19] * aa[4] * 0.03125;
        out[11] += Rt[55] * aa[4] * 0.09375;
        out[11] += Rt[62] * aa[5] * 0.015625;
        out[12] += Rt[0] * aa[2] * 0.125;
        out[12] += Rt[2] * aa[3] * 0.0625;
        out[12] += Rt[13] * aa[3] * 0.0625;
        out[12] += Rt[15] * aa[4] * 0.03125;
        out[12] += Rt[49] * aa[3] * 0.0625;
        out[12] += Rt[51] * aa[4] * 0.03125;
        out[12] += Rt[58] * aa[4] * 0.03125;
        out[12] += Rt[60] * aa[5] * 0.015625;
        out[13] += Rt[8] * aa[3] * 0.1875;
        out[13] += Rt[10] * aa[4] * 0.03125;
        out[13] += Rt[55] * aa[4] * 0.09375;
        out[13] += Rt[57] * aa[5] * 0.015625;
        out[14] += Rt[0] * aa[2] * 0.375;
        out[14] += Rt[2] * aa[3] * 0.375;
        out[14] += Rt[4] * aa[4] * 0.03125;
        out[14] += Rt[49] * aa[3] * 0.1875;
        out[14] += Rt[51] * aa[4] * 0.1875;
        out[14] += Rt[53] * aa[5] * 0.015625;
        out[15] += Rt[34] * aa[3] * 0.9375;
        out[15] += Rt[43] * aa[4] * 0.3125;
        out[15] += Rt[48] * aa[5] * 0.015625;
        out[16] += Rt[29] * aa[3] * 0.1875;
        out[16] += Rt[40] * aa[4] * 0.1875;
        out[16] += Rt[47] * aa[5] * 0.015625;
        out[17] += Rt[34] * aa[3] * 0.1875;
        out[17] += Rt[36] * aa[4] * 0.09375;
        out[17] += Rt[43] * aa[4] * 0.03125;
        out[17] += Rt[45] * aa[5] * 0.015625;
        out[18] += Rt[29] * aa[3] * 0.1875;
        out[18] += Rt[31] * aa[4] * 0.03125;
        out[18] += Rt[40] * aa[4] * 0.09375;
        out[18] += Rt[42] * aa[5] * 0.015625;
        out[19] += Rt[34] * aa[3] * 0.1875;
        out[19] += Rt[36] * aa[4] * 0.1875;
        out[19] += Rt[38] * aa[5] * 0.015625;
        out[20] += Rt[29] * aa[3] * 0.9375;
        out[20] += Rt[31] * aa[4] * 0.3125;
        out[20] += Rt[33] * aa[5] * 0.015625;
        out[21] += Rt[0] * aa[2] * 1.875;
        out[21] += Rt[13] * aa[3] * 2.8125;
        out[21] += Rt[22] * aa[4] * 0.46875;
        out[21] += Rt[27] * aa[5] * 0.015625;
        out[22] += Rt[8] * aa[3] * 0.9375;
        out[22] += Rt[19] * aa[4] * 0.3125;
        out[22] += Rt[26] * aa[5] * 0.015625;
        out[23] += Rt[0] * aa[2] * 0.375;
        out[23] += Rt[2] * aa[3] * 0.1875;
        out[23] += Rt[13] * aa[3] * 0.375;
        out[23] += Rt[15] * aa[4] * 0.1875;
        out[23] += Rt[22] * aa[4] * 0.03125;
        out[23] += Rt[24] * aa[5] * 0.015625;
        out[24] += Rt[8] * aa[3] * 0.5625;
        out[24] += Rt[10] * aa[4] * 0.09375;
        out[24] += Rt[19] * aa[4] * 0.09375;
        out[24] += Rt[21] * aa[5] * 0.015625;
        out[25] += Rt[0] * aa[2] * 0.375;
        out[25] += Rt[2] * aa[3] * 0.375;
        out[25] += Rt[4] * aa[4] * 0.03125;
        out[25] += Rt[13] * aa[3] * 0.1875;
        out[25] += Rt[15] * aa[4] * 0.1875;
        out[25] += Rt[17] * aa[5] * 0.015625;
        out[26] += Rt[8] * aa[3] * 0.9375;
        out[26] += Rt[10] * aa[4] * 0.3125;
        out[26] += Rt[12] * aa[5] * 0.015625;
        out[27] += Rt[0] * aa[2] * 1.875;
        out[27] += Rt[2] * aa[3] * 2.8125;
        out[27] += Rt[4] * aa[4] * 0.46875;
        out[27] += Rt[6] * aa[5] * 0.015625;
    }
}

template <int L> __device__ inline
void _dot_aux(double& out, double *Rt, double *auxvec,
              uint16_t *p1_ij, int nf3ij, int i, int nsp_per_block)
{
    if constexpr (L == 0) {
        out += Rt[p1_ij[0*nf3ij+i]*nsp_per_block] * auxvec[0];
    } else if constexpr (L == 1) {
        out += Rt[p1_ij[1*nf3ij+i]*nsp_per_block] * auxvec[1];
        out += Rt[p1_ij[2*nf3ij+i]*nsp_per_block] * auxvec[2];
        out += Rt[p1_ij[3*nf3ij+i]*nsp_per_block] * auxvec[3];
    } else if constexpr (L == 2) {
        out += Rt[p1_ij[0*nf3ij+i]*nsp_per_block] * auxvec[0];
        out += Rt[p1_ij[2*nf3ij+i]*nsp_per_block] * auxvec[2];
        out += Rt[p1_ij[4*nf3ij+i]*nsp_per_block] * auxvec[4];
        out += Rt[p1_ij[5*nf3ij+i]*nsp_per_block] * auxvec[5];
        out += Rt[p1_ij[7*nf3ij+i]*nsp_per_block] * auxvec[7];
        out += Rt[p1_ij[8*nf3ij+i]*nsp_per_block] * auxvec[8];
        out += Rt[p1_ij[9*nf3ij+i]*nsp_per_block] * auxvec[9];
    } else if constexpr (L == 3) {
        out += Rt[p1_ij[ 1*nf3ij+i]*nsp_per_block] * auxvec[1];
        out += Rt[p1_ij[ 3*nf3ij+i]*nsp_per_block] * auxvec[3];
        out += Rt[p1_ij[ 4*nf3ij+i]*nsp_per_block] * auxvec[4];
        out += Rt[p1_ij[ 6*nf3ij+i]*nsp_per_block] * auxvec[6];
        out += Rt[p1_ij[ 8*nf3ij+i]*nsp_per_block] * auxvec[8];
        out += Rt[p1_ij[ 9*nf3ij+i]*nsp_per_block] * auxvec[9];
        out += Rt[p1_ij[10*nf3ij+i]*nsp_per_block] * auxvec[10];
        out += Rt[p1_ij[12*nf3ij+i]*nsp_per_block] * auxvec[12];
        out += Rt[p1_ij[14*nf3ij+i]*nsp_per_block] * auxvec[14];
        out += Rt[p1_ij[15*nf3ij+i]*nsp_per_block] * auxvec[15];
        out += Rt[p1_ij[17*nf3ij+i]*nsp_per_block] * auxvec[17];
        out += Rt[p1_ij[18*nf3ij+i]*nsp_per_block] * auxvec[18];
        out += Rt[p1_ij[19*nf3ij+i]*nsp_per_block] * auxvec[19];
    } else if constexpr (L == 4) {
        out += Rt[p1_ij[ 0*nf3ij+i]*nsp_per_block] * auxvec[0];
        out += Rt[p1_ij[ 2*nf3ij+i]*nsp_per_block] * auxvec[2];
        out += Rt[p1_ij[ 4*nf3ij+i]*nsp_per_block] * auxvec[4];
        out += Rt[p1_ij[ 6*nf3ij+i]*nsp_per_block] * auxvec[6];
        out += Rt[p1_ij[ 8*nf3ij+i]*nsp_per_block] * auxvec[8];
        out += Rt[p1_ij[ 9*nf3ij+i]*nsp_per_block] * auxvec[9];
        out += Rt[p1_ij[11*nf3ij+i]*nsp_per_block] * auxvec[11];
        out += Rt[p1_ij[13*nf3ij+i]*nsp_per_block] * auxvec[13];
        out += Rt[p1_ij[14*nf3ij+i]*nsp_per_block] * auxvec[14];
        out += Rt[p1_ij[16*nf3ij+i]*nsp_per_block] * auxvec[16];
        out += Rt[p1_ij[18*nf3ij+i]*nsp_per_block] * auxvec[18];
        out += Rt[p1_ij[19*nf3ij+i]*nsp_per_block] * auxvec[19];
        out += Rt[p1_ij[21*nf3ij+i]*nsp_per_block] * auxvec[21];
        out += Rt[p1_ij[23*nf3ij+i]*nsp_per_block] * auxvec[23];
        out += Rt[p1_ij[24*nf3ij+i]*nsp_per_block] * auxvec[24];
        out += Rt[p1_ij[25*nf3ij+i]*nsp_per_block] * auxvec[25];
        out += Rt[p1_ij[27*nf3ij+i]*nsp_per_block] * auxvec[27];
        out += Rt[p1_ij[29*nf3ij+i]*nsp_per_block] * auxvec[29];
        out += Rt[p1_ij[30*nf3ij+i]*nsp_per_block] * auxvec[30];
        out += Rt[p1_ij[32*nf3ij+i]*nsp_per_block] * auxvec[32];
        out += Rt[p1_ij[33*nf3ij+i]*nsp_per_block] * auxvec[33];
        out += Rt[p1_ij[34*nf3ij+i]*nsp_per_block] * auxvec[34];
    } else if constexpr (L == 5) {
        out += Rt[p1_ij[ 1*nf3ij+i]*nsp_per_block] * auxvec[1];
        out += Rt[p1_ij[ 3*nf3ij+i]*nsp_per_block] * auxvec[3];
        out += Rt[p1_ij[ 5*nf3ij+i]*nsp_per_block] * auxvec[5];
        out += Rt[p1_ij[ 6*nf3ij+i]*nsp_per_block] * auxvec[6];
        out += Rt[p1_ij[ 8*nf3ij+i]*nsp_per_block] * auxvec[8];
        out += Rt[p1_ij[10*nf3ij+i]*nsp_per_block] * auxvec[10];
        out += Rt[p1_ij[12*nf3ij+i]*nsp_per_block] * auxvec[12];
        out += Rt[p1_ij[14*nf3ij+i]*nsp_per_block] * auxvec[14];
        out += Rt[p1_ij[15*nf3ij+i]*nsp_per_block] * auxvec[15];
        out += Rt[p1_ij[17*nf3ij+i]*nsp_per_block] * auxvec[17];
        out += Rt[p1_ij[19*nf3ij+i]*nsp_per_block] * auxvec[19];
        out += Rt[p1_ij[20*nf3ij+i]*nsp_per_block] * auxvec[20];
        out += Rt[p1_ij[21*nf3ij+i]*nsp_per_block] * auxvec[21];
        out += Rt[p1_ij[23*nf3ij+i]*nsp_per_block] * auxvec[23];
        out += Rt[p1_ij[25*nf3ij+i]*nsp_per_block] * auxvec[25];
        out += Rt[p1_ij[27*nf3ij+i]*nsp_per_block] * auxvec[27];
        out += Rt[p1_ij[29*nf3ij+i]*nsp_per_block] * auxvec[29];
        out += Rt[p1_ij[30*nf3ij+i]*nsp_per_block] * auxvec[30];
        out += Rt[p1_ij[32*nf3ij+i]*nsp_per_block] * auxvec[32];
        out += Rt[p1_ij[34*nf3ij+i]*nsp_per_block] * auxvec[34];
        out += Rt[p1_ij[35*nf3ij+i]*nsp_per_block] * auxvec[35];
        out += Rt[p1_ij[37*nf3ij+i]*nsp_per_block] * auxvec[37];
        out += Rt[p1_ij[39*nf3ij+i]*nsp_per_block] * auxvec[39];
        out += Rt[p1_ij[40*nf3ij+i]*nsp_per_block] * auxvec[40];
        out += Rt[p1_ij[42*nf3ij+i]*nsp_per_block] * auxvec[42];
        out += Rt[p1_ij[44*nf3ij+i]*nsp_per_block] * auxvec[44];
        out += Rt[p1_ij[45*nf3ij+i]*nsp_per_block] * auxvec[45];
        out += Rt[p1_ij[46*nf3ij+i]*nsp_per_block] * auxvec[46];
        out += Rt[p1_ij[48*nf3ij+i]*nsp_per_block] * auxvec[48];
        out += Rt[p1_ij[50*nf3ij+i]*nsp_per_block] * auxvec[50];
        out += Rt[p1_ij[51*nf3ij+i]*nsp_per_block] * auxvec[51];
        out += Rt[p1_ij[53*nf3ij+i]*nsp_per_block] * auxvec[53];
        out += Rt[p1_ij[54*nf3ij+i]*nsp_per_block] * auxvec[54];
        out += Rt[p1_ij[55*nf3ij+i]*nsp_per_block] * auxvec[55];
    } else if constexpr (L == 6) {
        out += Rt[p1_ij[ 0*nf3ij+i]*nsp_per_block] * auxvec[0];
        out += Rt[p1_ij[ 2*nf3ij+i]*nsp_per_block] * auxvec[2];
        out += Rt[p1_ij[ 4*nf3ij+i]*nsp_per_block] * auxvec[4];
        out += Rt[p1_ij[ 6*nf3ij+i]*nsp_per_block] * auxvec[6];
        out += Rt[p1_ij[ 8*nf3ij+i]*nsp_per_block] * auxvec[8];
        out += Rt[p1_ij[10*nf3ij+i]*nsp_per_block] * auxvec[10];
        out += Rt[p1_ij[12*nf3ij+i]*nsp_per_block] * auxvec[12];
        out += Rt[p1_ij[13*nf3ij+i]*nsp_per_block] * auxvec[13];
        out += Rt[p1_ij[15*nf3ij+i]*nsp_per_block] * auxvec[15];
        out += Rt[p1_ij[17*nf3ij+i]*nsp_per_block] * auxvec[17];
        out += Rt[p1_ij[19*nf3ij+i]*nsp_per_block] * auxvec[19];
        out += Rt[p1_ij[21*nf3ij+i]*nsp_per_block] * auxvec[21];
        out += Rt[p1_ij[22*nf3ij+i]*nsp_per_block] * auxvec[22];
        out += Rt[p1_ij[24*nf3ij+i]*nsp_per_block] * auxvec[24];
        out += Rt[p1_ij[26*nf3ij+i]*nsp_per_block] * auxvec[26];
        out += Rt[p1_ij[27*nf3ij+i]*nsp_per_block] * auxvec[27];
        out += Rt[p1_ij[29*nf3ij+i]*nsp_per_block] * auxvec[29];
        out += Rt[p1_ij[31*nf3ij+i]*nsp_per_block] * auxvec[31];
        out += Rt[p1_ij[33*nf3ij+i]*nsp_per_block] * auxvec[33];
        out += Rt[p1_ij[34*nf3ij+i]*nsp_per_block] * auxvec[34];
        out += Rt[p1_ij[36*nf3ij+i]*nsp_per_block] * auxvec[36];
        out += Rt[p1_ij[38*nf3ij+i]*nsp_per_block] * auxvec[38];
        out += Rt[p1_ij[40*nf3ij+i]*nsp_per_block] * auxvec[40];
        out += Rt[p1_ij[42*nf3ij+i]*nsp_per_block] * auxvec[42];
        out += Rt[p1_ij[43*nf3ij+i]*nsp_per_block] * auxvec[43];
        out += Rt[p1_ij[45*nf3ij+i]*nsp_per_block] * auxvec[45];
        out += Rt[p1_ij[47*nf3ij+i]*nsp_per_block] * auxvec[47];
        out += Rt[p1_ij[48*nf3ij+i]*nsp_per_block] * auxvec[48];
        out += Rt[p1_ij[49*nf3ij+i]*nsp_per_block] * auxvec[49];
        out += Rt[p1_ij[51*nf3ij+i]*nsp_per_block] * auxvec[51];
        out += Rt[p1_ij[53*nf3ij+i]*nsp_per_block] * auxvec[53];
        out += Rt[p1_ij[55*nf3ij+i]*nsp_per_block] * auxvec[55];
        out += Rt[p1_ij[57*nf3ij+i]*nsp_per_block] * auxvec[57];
        out += Rt[p1_ij[58*nf3ij+i]*nsp_per_block] * auxvec[58];
        out += Rt[p1_ij[60*nf3ij+i]*nsp_per_block] * auxvec[60];
        out += Rt[p1_ij[62*nf3ij+i]*nsp_per_block] * auxvec[62];
        out += Rt[p1_ij[63*nf3ij+i]*nsp_per_block] * auxvec[63];
        out += Rt[p1_ij[65*nf3ij+i]*nsp_per_block] * auxvec[65];
        out += Rt[p1_ij[67*nf3ij+i]*nsp_per_block] * auxvec[67];
        out += Rt[p1_ij[68*nf3ij+i]*nsp_per_block] * auxvec[68];
        out += Rt[p1_ij[70*nf3ij+i]*nsp_per_block] * auxvec[70];
        out += Rt[p1_ij[72*nf3ij+i]*nsp_per_block] * auxvec[72];
        out += Rt[p1_ij[73*nf3ij+i]*nsp_per_block] * auxvec[73];
        out += Rt[p1_ij[74*nf3ij+i]*nsp_per_block] * auxvec[74];
        out += Rt[p1_ij[76*nf3ij+i]*nsp_per_block] * auxvec[76];
        out += Rt[p1_ij[78*nf3ij+i]*nsp_per_block] * auxvec[78];
        out += Rt[p1_ij[79*nf3ij+i]*nsp_per_block] * auxvec[79];
        out += Rt[p1_ij[81*nf3ij+i]*nsp_per_block] * auxvec[81];
        out += Rt[p1_ij[82*nf3ij+i]*nsp_per_block] * auxvec[82];
        out += Rt[p1_ij[83*nf3ij+i]*nsp_per_block] * auxvec[83];
    }
}

template <int LK> __device__ inline
void unrolled_contract_int3c2e(RysIntEnvVars envs, JKMatrix jk,
                               int *shl_pair_offsets, uint32_t *bas_ij_idx,
                               int *pair_ij_loc, int *nsp_lookup)
{
    constexpr int lk = LK;
    constexpr int nfk = (lk + 1) * (lk + 2) / 2;
    constexpr int nf3k = nfk * (lk + 3) / 3;
    int sp_block_id = gridDim.y - blockIdx.y - 1;
    int ksh = gridDim.x - blockIdx.x - 1 + envs.nbas;
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
    }
    __syncthreads();
    int bas_ij0 = bas_ij_idx[shl_pair0];
    int ish0 = bas_ij0 / envs.nbas;
    int jsh0 = bas_ij0 % envs.nbas;
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lij = li + lj;
    __shared__ int order, nf3ij, nf3ijk, kprim;
    __shared__ int nsp_per_block, Rt_stride;
    __shared__ double rk[3];
    if (thread_id == 0) {
        order = lij + lk;
        nf3ij = (lij+1)*(lij+2)*(lij+3) / 6;
        nf3ijk = (order+1)*(order+2)*(order+3) / 6;
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        int rk_ptr = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        rk[0] = env[rk_ptr+0];
        rk[1] = env[rk_ptr+1];
        rk[2] = env[rk_ptr+2];
        nsp_per_block = nsp_lookup[lij*(L_AUX_MAX+1)+lk];
        Rt_stride = blockDim.x / nsp_per_block;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int Rt_id = thread_id / nsp_per_block;

    extern __shared__ double shared_memory[];
    double *gamma_inc = shared_memory + sp_id;
    double *Rt_buf = shared_memory + (order+1) * nsp_per_block;
    uint16_t *p1_ij = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lk];
    int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lk];
    for (int kp = 0; kp < kprim; ++kp) {
        __syncthreads();
        __shared__ double ak, ck;
        if (thread_id == 0) {
            ck = env[bas[ksh*BAS_SLOTS+PTR_COEFF] + kp] * PI_FAC;
            ak = env[bas[ksh*BAS_SLOTS+PTR_EXP] + kp];
        }
        double vj_xyz[nf3k];
#pragma unroll
        for (int n = 0; n < nf3k; ++n) {
            vj_xyz[n] = 0;
        }
        for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
            __syncthreads();
            int bas_ij;
            if (pair_ij < shl_pair1) {
                bas_ij = bas_ij_idx[pair_ij];
            } else {
                bas_ij = bas_ij_idx[shl_pair0];
            }
            int ish = bas_ij / envs.nbas;
            int jsh = bas_ij % envs.nbas;
            double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
            double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double aij = ai + aj;
            double xij = (ai * ri[0] + aj * rj[0]) / aij;
            double yij = (ai * ri[1] + aj * rj[1]) / aij;
            double zij = (ai * ri[2] + aj * rj[2]) / aij;
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * ak / (aij + ak);
            double *Rt, *buf;
            if (order % 2 == 0) {
                Rt = Rt_buf + sp_id;
                buf = Rt + nf3ijk * nsp_per_block;
            } else {
                buf = Rt_buf + sp_id;
                Rt = buf + nf3ijk * nsp_per_block;
            }
            if (Rt_id == 0) {
                double fac = ck/(aij*ak*sqrt(aij+ak));
                if (pair_ij >= shl_pair1) {
                    fac = 0;
                }
                boys_fn(gamma_inc, theta, rr, jk.omega, fac, order, 0, nsp_per_block);
                Rt[0] = gamma_inc[order*nsp_per_block];
            }
            for (int n = 1; n <= order; ++n) {
                __syncthreads();
                // swap input and output
                double *tmp = buf;
                buf = Rt;
                Rt = tmp;
                if (n == 1) {
                    if (Rt_id == 0) {
                        double _Rt_0 = buf[0];
                        Rt[1*nsp_per_block] = zpq * _Rt_0;
                        Rt[2*nsp_per_block] = ypq * _Rt_0;
                        Rt[3*nsp_per_block] = xpq * _Rt_0;
                        Rt[0] = gamma_inc[(order-n)*nsp_per_block];
                    }
                } else if (n == 2) {
                    if (Rt_id == 0) {
                        double _Rt_0 = buf[0];
                        double _Rt_1 = buf[1*nsp_per_block];
                        double _Rt_2 = buf[2*nsp_per_block];
                        double _Rt_3 = buf[3*nsp_per_block];
                        Rt[1*nsp_per_block] = zpq * _Rt_0;
                        Rt[2*nsp_per_block] = zpq * _Rt_1 + _Rt_0;
                        Rt[3*nsp_per_block] = ypq * _Rt_0;
                        Rt[4*nsp_per_block] = ypq * _Rt_1;
                        Rt[5*nsp_per_block] = ypq * _Rt_2 + _Rt_0;
                        Rt[6*nsp_per_block] = xpq * _Rt_0;
                        Rt[7*nsp_per_block] = xpq * _Rt_1;
                        Rt[8*nsp_per_block] = xpq * _Rt_2;
                        Rt[9*nsp_per_block] = xpq * _Rt_3 + _Rt_0;
                        Rt[0] = gamma_inc[(order-n)*nsp_per_block];
                    }
                } else {
                    iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsp_per_block, Rt_id, Rt_stride);
                    if (Rt_id == 0) {
                        Rt[0] = gamma_inc[(order-n)*nsp_per_block];
                    }
                }
            }
            __syncthreads();

            if (pair_ij < shl_pair1) {
                int ij_loc0 = pair_ij_loc[pair_ij];
                double *dm = jk.dm + ij_loc0;
                for (int i = Rt_id; i < nf3ij; i += Rt_stride) {
                    double dm_ij = dm[i];
#pragma unroll
                    for (int k = 0; k < nf3k; k++) {
                        int off = k * nf3ij;
                        double s = Rt[p1_ij[off+i]*nsp_per_block];
                        vj_xyz[k] += s * dm_ij;
                    }
                }
            }
        }
#pragma unroll
        for (int k = 0; k < nf3k; k++) {
            vj_xyz[k] *= efg_phase[k];
        }

        __syncthreads();
        double vj_aux[nfk];
#pragma unroll
        for (int n = 0; n < nfk; ++n) {
            vj_aux[n] = 0;
        }
        _dot_Et<LK>(vj_aux, vj_xyz, ak);
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh] - ao_loc[envs.nbas];
        int lane = thread_id % warpSize;
        int wid  = thread_id / warpSize;
#pragma unroll
        for (int k = 0; k < nfk; k++) {
            double val = vj_aux[k];
            for (int offset = warpSize/2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane == 0) {
                shared_memory[wid] = val;
            }
            __syncthreads();

            if (thread_id < 8) {
                val = shared_memory[lane];
            }
            for (int offset = 4; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xff, val, offset);
            }
            if (thread_id == 0) {
                atomicAdd(jk.vj+k0+k, val);
            }
            __syncthreads();
        }
    }
}

__global__ static
void contract_int3c2e_kernel(RysIntEnvVars envs, JKMatrix jk,
                             int *shl_pair_offsets, uint32_t *bas_ij_idx,
                             int *pair_ij_loc, int *nsp_lookup)
{
    int ksh = gridDim.x - blockIdx.x - 1 + envs.nbas;
    int lk = envs.bas[ANG_OF + ksh*BAS_SLOTS];
    switch (lk) {
    case 0: unrolled_contract_int3c2e<0>(envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup); break;
    case 1: unrolled_contract_int3c2e<1>(envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup); break;
    case 2: unrolled_contract_int3c2e<2>(envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup); break;
    case 3: unrolled_contract_int3c2e<3>(envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup); break;
    case 4: unrolled_contract_int3c2e<4>(envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup); break;
    case 5: unrolled_contract_int3c2e<5>(envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup); break;
    case 6: unrolled_contract_int3c2e<6>(envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup); break;
    }
}

//__global__ static
template <int LK> __device__ inline
void unroll_contract_auxvec(RysIntEnvVars envs, JKMatrix jk,
                            int *shl_pair_offsets, int *ksh_offsets,
                            uint32_t *bas_ij_idx, int *pair_ij_loc,
                            int *aux_loc, int *nsp_lookup)
{
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1;
    if (thread_id == 0) {
        int sp_block_id = gridDim.x - blockIdx.x - 1;
        int ksh_block_id = gridDim.y - blockIdx.y - 1;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
    }
    __syncthreads();
    int bas_ij0 = bas_ij_idx[shl_pair0];
    int ish0 = bas_ij0 / envs.nbas;
    int jsh0 = bas_ij0 % envs.nbas;
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    //int lk = bas[ksh0*BAS_SLOTS+ANG_OF];
    constexpr int lk = LK;
    int lij = li + lj;
    int order = lij + lk;
    constexpr int nfk = (lk + 1) * (lk + 2) / 2;
    constexpr int nf3k = nfk * (lk + 3) / 3;
    int nf3ij = (lij+1)*(lij+2)*(lij+3) / 6;
    int nf3ijk = (order+1)*(order+2)*(order+3) / 6;
    __shared__ int nsp_per_block, Rt_stride;
    if (thread_id == 0) {
        nsp_per_block = nsp_lookup[lij*(L_AUX_MAX+1)+lk];
        Rt_stride = blockDim.x / nsp_per_block;
    }
    __syncthreads();
    int sp_id = thread_id % nsp_per_block;
    int Rt_id = thread_id / nsp_per_block;
    extern __shared__ double shared_memory[];
    double *gamma_inc = shared_memory + sp_id;
    double *auxvec_cache = shared_memory + (order+1) * nsp_per_block;
    double *Rt_buf = auxvec_cache + nf3k;
    uint16_t *p1_ij = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lk];
    int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lk];
    double *auxvec = jk.dm;

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double vj_xyz[IJ_SIZE];
#pragma unroll
        for (int n = 0; n < IJ_SIZE; ++n) {
            vj_xyz[n] = 0;
        }
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / envs.nbas;
        int jsh = bas_ij % envs.nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double aij = ai + aj;
        double xij = (ai * ri[0] + aj * rj[0]) / aij;
        double yij = (ai * ri[1] + aj * rj[1]) / aij;
        double zij = (ai * ri[2] + aj * rj[2]) / aij;
        for (int ksh = ksh0; ksh < ksh1; ++ksh) {
            __syncthreads();
            int k_loc0 = aux_loc[ksh - envs.nbas];
            if (thread_id < nf3k) {
                auxvec_cache[thread_id] = efg_phase[thread_id] *
                    auxvec[k_loc0+thread_id];
            }
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            double ak = env[expk];
            double theta = aij * ak / (aij + ak);
            double *Rt, *buf;
            if (order % 2 == 0) {
                Rt = Rt_buf + sp_id;
                buf = Rt + nf3ijk * nsp_per_block;
            } else {
                buf = Rt_buf + sp_id;
                Rt = buf + nf3ijk * nsp_per_block;
            }
            if (Rt_id == 0) {
                double fac = PI_FAC/(aij*ak*sqrt(aij+ak));
                if (pair_ij >= shl_pair1) {
                    fac = 0;
                }
                boys_fn(gamma_inc, theta, rr, jk.omega, fac, order, 0, nsp_per_block);
                Rt[0] = gamma_inc[order*nsp_per_block];
            }
            for (int n = 1; n <= order; ++n) {
                __syncthreads();
                // swap input and output
                double *tmp = buf;
                buf = Rt;
                Rt = tmp;
                if (n == 1) {
                    if (Rt_id == 0) {
                        double _Rt_0 = buf[0];
                        Rt[1*nsp_per_block] = zpq * _Rt_0;
                        Rt[2*nsp_per_block] = ypq * _Rt_0;
                        Rt[3*nsp_per_block] = xpq * _Rt_0;
                        Rt[0] = gamma_inc[(order-n)*nsp_per_block];
                    }
                } else if (n == 2) {
                    if (Rt_id == 0) {
                        double _Rt_0 = buf[0];
                        double _Rt_1 = buf[1*nsp_per_block];
                        double _Rt_2 = buf[2*nsp_per_block];
                        double _Rt_3 = buf[3*nsp_per_block];
                        Rt[1*nsp_per_block] = zpq * _Rt_0;
                        Rt[2*nsp_per_block] = zpq * _Rt_1 + _Rt_0;
                        Rt[3*nsp_per_block] = ypq * _Rt_0;
                        Rt[4*nsp_per_block] = ypq * _Rt_1;
                        Rt[5*nsp_per_block] = ypq * _Rt_2 + _Rt_0;
                        Rt[6*nsp_per_block] = xpq * _Rt_0;
                        Rt[7*nsp_per_block] = xpq * _Rt_1;
                        Rt[8*nsp_per_block] = xpq * _Rt_2;
                        Rt[9*nsp_per_block] = xpq * _Rt_3 + _Rt_0;
                        Rt[0] = gamma_inc[(order-n)*nsp_per_block];
                    }
                } else {
                    iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsp_per_block, Rt_id, Rt_stride);
                    if (Rt_id == 0) {
                        Rt[0] = gamma_inc[(order-n)*nsp_per_block];
                    }
                }
            }
            __syncthreads();
            if (pair_ij < shl_pair1) {
#pragma unroll
                for (int n = 0, i = Rt_id; n < IJ_SIZE; ++n, i += Rt_stride) {
                    if (i >= nf3ij) break;
                    _dot_aux<LK>(vj_xyz[n], Rt, auxvec_cache, p1_ij, nf3ij, i,
                                 nsp_per_block);
                }
            }
        }
        if (pair_ij < shl_pair1) {
#pragma unroll
            for (int n = 0, i = Rt_id; n < IJ_SIZE; ++n, i += Rt_stride) {
                if (i >= nf3ij) break;
                int ij_loc0 = pair_ij_loc[pair_ij];
                atomicAdd(jk.vj+ij_loc0+i, vj_xyz[n]);
            }
        }
    }
}

__global__ static
void contract_auxvec_kernel(RysIntEnvVars envs, JKMatrix jk,
                            int *shl_pair_offsets, int *ksh_offsets,
                            uint32_t *bas_ij_idx, int *pair_ij_loc,
                            int *aux_loc, int *nsp_lookup)
{
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int ksh = ksh_offsets[ksh_block_id];
    int lk = envs.bas[ANG_OF + ksh*BAS_SLOTS];
    switch (lk) {
    case 0: unroll_contract_auxvec<0>(envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc, aux_loc, nsp_lookup); break;
    case 1: unroll_contract_auxvec<1>(envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc, aux_loc, nsp_lookup); break;
    case 2: unroll_contract_auxvec<2>(envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc, aux_loc, nsp_lookup); break;
    case 3: unroll_contract_auxvec<3>(envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc, aux_loc, nsp_lookup); break;
    case 4: unroll_contract_auxvec<4>(envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc, aux_loc, nsp_lookup); break;
    case 5: unroll_contract_auxvec<5>(envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc, aux_loc, nsp_lookup); break;
    case 6: unroll_contract_auxvec<6>(envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc, aux_loc, nsp_lookup); break;
    }
}

extern "C" {
// contract('ijP,ji->P', int3c2e, dm)
int contract_int3c2e_dm(double *vj, double *dm, int n_dm, int naux,
                        RysIntEnvVars *envs, int shm_size,
                        int nbatches_shl_pair, int nksh,
                        int *shl_pair_offsets, uint32_t *bas_ij_idx,
                        int *pair_ij_loc, int *nsp_lookup, double omega)
{
    assert(n_dm == 1);
    cudaFuncSetAttribute(contract_int3c2e_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    JKMatrix jk = {vj, NULL, dm, n_dm, 0, omega};
    dim3 threads(THREADS);
    dim3 blocks(nksh, nbatches_shl_pair);
    contract_int3c2e_kernel<<<blocks, threads, shm_size>>>(
        *envs, jk, shl_pair_offsets, bas_ij_idx, pair_ij_loc, nsp_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm, error message = %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

// contract('ijP,P->ij', int3c2e, auxvec)
int contract_int3c2e_auxvec(double *vj, double *auxvec, int n_dm, int naux,
                            RysIntEnvVars *envs, int shm_size,
                            int nbatches_shl_pair, int nbatches_ksh,
                            int *shl_pair_offsets, int *ksh_offsets,
                            uint32_t *bas_ij_idx, int *pair_ij_loc, int *aux_loc,
                            int *nsp_lookup, double omega)
{
    assert(n_dm == 1);
    JKMatrix jk = {vj, NULL, auxvec, n_dm, 0, omega};
    dim3 threads(THREADS);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    contract_auxvec_kernel<<<blocks, threads, shm_size>>>(
        *envs, jk, shl_pair_offsets, ksh_offsets, bas_ij_idx, pair_ij_loc,
        aux_loc, nsp_lookup);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_auxvec, error message = %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
