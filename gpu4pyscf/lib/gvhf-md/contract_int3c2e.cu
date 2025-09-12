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

template <int LK> __device__ inline
void unrolled_contract_int3c2e(Int3c2eEnvVars envs, JKMatrix jk, BDiv3c2eBounds bounds,
                               int *pair_ij_loc)
{
    int sp_block_id = gridDim.y - blockIdx.y - 1;
    int ksh = blockIdx.x + envs.nbas;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    double ck = env[bas[ksh*BAS_SLOTS+PTR_COEFF]] * PI_FAC;
    double ak = env[bas[ksh*BAS_SLOTS+PTR_EXP]];
    double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
    double xk = rk[0];
    double yk = rk[1];
    double zk = rk[2];
    constexpr int lk = LK;
    constexpr int nfk = (lk + 1) * (lk + 2) / 2;

    int lij = li + lj;
    int order = lij + lk;
    int nf3ij = (lij+1)*(lij+2)*(lij+3) / 6;
    int nf3ijkl = (order+1)*(order+2)*(order+3) / 6;
    double *dm = jk.dm;
    int *nsp_lookup = bounds.nst_lookup;
    register int nsp_per_block = nsp_lookup[lij*(L_AUX_MAX+1)+lk];
    int thread_id = threadIdx.x;
    int Rt_stride = blockDim.x / nsp_per_block;
    int sp_id = thread_id % nsp_per_block;
    int Rt_id = thread_id / nsp_per_block;

    extern __shared__ double phase[];
    double *gamma_inc = phase + nfk;
    double *Rt_buf = phase + nfk + (order+1) * nsp_per_block;
    uint16_t *p1_ij = Rt2_kl_ij + Rt2_idx_offsets[lij*RT2_MAX+lk];
    int8_t *efg_phase = c_Rt2_efg_phase + Rt2_idx_offsets[lk];
    double vj_aux[nfk];
#pragma unroll
    for (int n = 0; n < nfk; ++n) {
        vj_aux[n] = 0;
    }
    if (thread_id < nfk) {
        phase[thread_id] = efg_phase[thread_id];
    }

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        __syncthreads();
        int bas_ij, ij_loc0;
        if (pair_ij < shl_pair1) {
            bas_ij = bounds.bas_ij_idx[pair_ij];
            ij_loc0 = pair_ij_loc[pair_ij];
        } else {
            bas_ij = bounds.bas_ij_idx[shl_pair0];
            ij_loc0 = pair_ij_loc[shl_pair0];
            ck = 0;
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
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
        double xpq = xij - xk;
        double ypq = yij - yk;
        double zpq = zij - zk;
        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
        double theta = aij * ak / (aij + ak);
        if (Rt_id == 0) {
            boys_fn(gamma_inc, theta, rr, jk.omega, ck/(aij*ak*sqrt(aij+ak)),
                    order, sp_id, nsp_per_block);
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

        Rt = Rt_buf;
        for (int i = Rt_id; i < nf3ij; i += Rt_stride) {
            double dm_ij = dm[ij_loc0+i];
#pragma unroll
            for (int k = 0; k < nfk; k++) {
                int off = k * nf3ij;
                double s = Rt[sp_id+p1_ij[off+i]*nsp_per_block];
                vj_aux[k] += phase[k] * s * dm_ij;
            }
        }
    }

    int *ao_loc = envs.ao_loc;
    int k0 = ao_loc[ksh] - ao_loc[bounds.aux_sh_offset];
    double *vj = jk.vj + k0;
    typedef cub::BlockReduce<double, 256> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
#pragma unroll
    for (int k = 0; k < nfk; k++) {
        double vj_aux_k = BlockReduceT(temp_storage).Sum(vj_aux[k]);
        atomicAdd(vj+k, vj_aux_k);
    }
}

__global__ static
void contract_int3c2e_kernel(Int3c2eEnvVars envs, JKMatrix jk, BDiv3c2eBounds bounds,
                             int *pair_ij_loc)
{
    int ksh = blockIdx.x + envs.nbas;
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

    dim3 threads(256);
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
