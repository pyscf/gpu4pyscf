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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "gint-rys/int3c2e.cuh"
#include "gvhf-rys/vhf.cuh"
#include "gvhf-md/boys.cu"
#include "gvhf-md/md_j.cuh"

#define RT2_MAX 9
#define THREADS 256

extern __constant__ uint16_t c_Rt_idx[];
extern __constant__ int8_t c_Rt_tuv_fac[];
extern __constant__ int8_t c_Rt2_efg_phase[];
extern __device__ int Rt2_idx_offsets[];
extern __device__ uint16_t Rt2_kl_ij[];

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
    if (L == 0) {
        out[0] += Rt[0];
    } else if (L == 1) {
        out[0] += Rt[3] * aa[0] * 0.5;
        out[1] += Rt[2] * aa[0] * 0.5;
        out[2] += Rt[1] * aa[0] * 0.5;
    } else if (L == 2) {
        out[0] += Rt[0] * aa[0] * 0.5;
        out[0] += Rt[9] * aa[1] * 0.25;
        out[1] += Rt[8] * aa[1] * 0.25;
        out[2] += Rt[7] * aa[1] * 0.25;
        out[3] += Rt[0] * aa[0] * 0.5;
        out[3] += Rt[5] * aa[1] * 0.25;
        out[4] += Rt[4] * aa[1] * 0.25;
        out[5] += Rt[0] * aa[0] * 0.5;
        out[5] += Rt[2] * aa[1] * 0.25;
    } else if (L == 3) {
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
    } else if (L == 4) {
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
    } else if (L == 5) {
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
    } else if (L == 6) {
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

template <int LK> __device__ inline
void unrolled_contract_int3c2e(Int3c2eEnvVars envs, JKMatrix jk, BDiv3c2eBounds bounds,
                               int *pair_ij_loc)
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
        shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
        shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    }
    __syncthreads();
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int ish0 = bas_ij0 / envs.nbas;
    int jsh0 = bas_ij0 % envs.nbas;
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lij = li + lj;
    __shared__ int order, nf3ij, nf3ijkl, kprim;
    __shared__ double rk[3];
    if (thread_id == 0) {
        order = lij + lk;
        nf3ij = (lij+1)*(lij+2)*(lij+3) / 6;
        nf3ijkl = (order+1)*(order+2)*(order+3) / 6;
        kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
        int rk_ptr = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        rk[0] = env[rk_ptr+0];
        rk[1] = env[rk_ptr+1];
        rk[2] = env[rk_ptr+2];
    }
    __syncthreads();
    int *nsp_lookup = bounds.nst_lookup;
    int nsp_per_block = nsp_lookup[lij*(L_AUX_MAX+1)+lk];
    int Rt_stride = blockDim.x / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;
    int Rt_id = thread_id / nsp_per_block;

    extern __shared__ double phase[];
    double *gamma_inc = phase + nf3k;
    double *Rt_buf = phase + nf3k + (order+1) * nsp_per_block;
    uint16_t *p1_ij = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lk];
    int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lk];
    if (thread_id < nf3k) {
        phase[thread_id] = efg_phase[thread_id];
    }
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
                bas_ij = bounds.bas_ij_idx[pair_ij];
            } else {
                bas_ij = bounds.bas_ij_idx[shl_pair0];
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
            double *Rt, *buf;
            if (order % 2 == 0) {
                Rt = Rt_buf + sp_id;
                buf = Rt + nf3ijkl * nsp_per_block;
            } else {
                buf = Rt_buf + sp_id;
                Rt = buf + nf3ijkl * nsp_per_block;
            }
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq*xpq + ypq*ypq + zpq*zpq;
            double theta = aij * ak / (aij + ak);
            if (Rt_id == 0) {
                double fac = ck/(aij*ak*sqrt(aij+ak));
                if (pair_ij >= shl_pair1) {
                    fac = 0;
                }
                boys_fn(gamma_inc, theta, rr, jk.omega, fac, order, sp_id, nsp_per_block);
                Rt[0] = gamma_inc[sp_id+order*nsp_per_block];
            }
            for (int n = 1; n <= order; ++n) {
                __syncthreads();
                // swap input and output
                double *tmp = buf;
                buf = Rt;
                Rt = tmp;
                if (Rt_id == 0) {
                    Rt[0] = gamma_inc[sp_id+(order-n)*nsp_per_block];
                }
                if (n == 1) {
                    if (Rt_id == 0) {
                        Rt[1*nsp_per_block] = zpq * buf[0*nsp_per_block];
                        Rt[2*nsp_per_block] = ypq * buf[0*nsp_per_block];
                        Rt[3*nsp_per_block] = xpq * buf[0*nsp_per_block];
                    }
                } else if (n == 2) {
                    if (Rt_id == 0) {
                        Rt[1*nsp_per_block] = zpq * buf[0*nsp_per_block];
                        Rt[2*nsp_per_block] = zpq * buf[1*nsp_per_block] + buf[0*nsp_per_block];
                        Rt[3*nsp_per_block] = ypq * buf[0*nsp_per_block];
                        Rt[4*nsp_per_block] = ypq * buf[1*nsp_per_block];
                        Rt[5*nsp_per_block] = ypq * buf[2*nsp_per_block] + buf[0*nsp_per_block];
                        Rt[6*nsp_per_block] = xpq * buf[0*nsp_per_block];
                        Rt[7*nsp_per_block] = xpq * buf[1*nsp_per_block];
                        Rt[8*nsp_per_block] = xpq * buf[2*nsp_per_block];
                        Rt[9*nsp_per_block] = xpq * buf[3*nsp_per_block] + buf[0*nsp_per_block];
                    }
                } else {
                    iter_Rt_n(Rt, buf, xpq, ypq, zpq, n, nsp_per_block, Rt_id, Rt_stride);
                }
            }
            __syncthreads();

            if (pair_ij < shl_pair1) {
                int ij_loc0 = pair_ij_loc[pair_ij];
                double *dm = jk.dm + ij_loc0;
                Rt = Rt_buf;
                for (int i = Rt_id; i < nf3ij; i += Rt_stride) {
                    double dm_ij = dm[i];
#pragma unroll
                    for (int k = 0; k < nf3k; k++) {
                        int off = k * nf3ij;
                        double s = Rt[sp_id+p1_ij[off+i]*nsp_per_block];
                        vj_xyz[k] += phase[k] * s * dm_ij;
                    }
                }
            }
        }

        double vj_aux[nfk];
#pragma unroll
        for (int n = 0; n < nfk; ++n) {
            vj_aux[n] = 0;
        }
        _dot_Et<LK>(vj_aux, vj_xyz, ak);
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh] - ao_loc[bounds.aux_sh_offset];
        double *vj = jk.vj + k0;
        typedef cub::BlockReduce<double, THREADS> BlockReduceT;
        __shared__ typename BlockReduceT::TempStorage temp_storage;
#pragma unroll
        for (int k = 0; k < nfk; k++) {
            double sum_jaux = BlockReduceT(temp_storage).Sum(vj_aux[k]);
            if (thread_id == 0) {
                atomicAdd(vj+k, sum_jaux);
            }
            __syncthreads();
        }
    }
}

__global__ static
void contract_int3c2e_kernel(Int3c2eEnvVars envs, JKMatrix jk, BDiv3c2eBounds bounds,
                             int *pair_ij_loc)
{
    int ksh = gridDim.x - blockIdx.x - 1 + envs.nbas;
    int lk = envs.bas[ANG_OF + ksh*BAS_SLOTS];
    switch (lk) {
    case 0: unrolled_contract_int3c2e<0>(envs, jk, bounds, pair_ij_loc); break;
    case 1: unrolled_contract_int3c2e<1>(envs, jk, bounds, pair_ij_loc); break;
    case 2: unrolled_contract_int3c2e<2>(envs, jk, bounds, pair_ij_loc); break;
    case 3: unrolled_contract_int3c2e<3>(envs, jk, bounds, pair_ij_loc); break;
    case 4: unrolled_contract_int3c2e<4>(envs, jk, bounds, pair_ij_loc); break;
    case 5: unrolled_contract_int3c2e<5>(envs, jk, bounds, pair_ij_loc); break;
    case 6: unrolled_contract_int3c2e<6>(envs, jk, bounds, pair_ij_loc); break;
    }
}

extern "C" {
// contract('ijP,ji->P', int3c2e, dm)
int contract_int3c2e_dm(double *vj, double *dm, int n_dm, int naux,
                        Int3c2eEnvVars *envs, int shm_size,
                        int nbatches_shl_pair, int nksh,
                        int *shl_pair_offsets, int *bas_ij_idx,
                        int *pair_ij_loc, int *nsp_lookup,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
    BDiv3c2eBounds bounds = {naux, nbas, bas_ij_idx, shl_pair_offsets,
        NULL, NULL, nsp_lookup};

    double omega = env[PTR_RANGE_OMEGA];
    JKMatrix jk = {vj, NULL, dm, 1, 0, omega};

    dim3 threads(THREADS);
    dim3 blocks(nksh, nbatches_shl_pair);
    contract_int3c2e_kernel<<<blocks, threads, shm_size>>>(*envs, jk, bounds, pair_ij_loc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in contract_int3c2e_dm, error message = %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int MD_int3c2e_init(int shm_size)
{
    cudaFuncSetAttribute(contract_int3c2e_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    return 0;
}
}
