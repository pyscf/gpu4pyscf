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
#include <cuda_runtime.h>

#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "int3c2e.cuh"

__device__ static
void int3c2e_bdiv_000(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 1;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(1, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(1, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 2*nst_per_block;
                rys_roots(1, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(1, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                gout0 += 1 * fac1 * wt;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 1 * naux + ksh_in_block * 1;
        eri_tensor[0*naux + 0] = gout0;
    }
}

__device__ static
void int3c2e_bdiv_100(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 1;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(1, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(1, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 2*nst_per_block;
                rys_roots(1, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(1, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                gout0 += trr_10x * fac1 * wt;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += 1 * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += 1 * fac1 * trr_10z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 3 * naux + ksh_in_block * 1;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[2*naux + 0] = gout2;
    }
}

__device__ static
void int3c2e_bdiv_110(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                gout0 += hrr_110x * fac1 * wt;
                double hrr_010x = trr_10x - xjxi * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += hrr_010x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += hrr_010x * fac1 * trr_10z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout3 += trr_10x * hrr_010y * wt;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout4 += 1 * hrr_110y * wt;
                gout5 += 1 * hrr_010y * trr_10z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout6 += trr_10x * fac1 * hrr_010z;
                gout7 += 1 * trr_10y * hrr_010z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout8 += 1 * fac1 * hrr_110z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 9 * naux + ksh_in_block * 1;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[6*naux + 0] = gout6;
        eri_tensor[7*naux + 0] = gout7;
        eri_tensor[8*naux + 0] = gout8;
    }
}

__device__ static
void int3c2e_bdiv_200(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                gout0 += trr_20x * fac1 * wt;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += trr_10x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += trr_10x * fac1 * trr_10z;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout3 += 1 * trr_20y * wt;
                gout4 += 1 * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout5 += 1 * fac1 * trr_20z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 6 * naux + ksh_in_block * 1;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[5*naux + 0] = gout5;
    }
}

__device__ static
void int3c2e_bdiv_210(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                double hrr_210x = trr_30x - xjxi * trr_20x;
                gout0 += hrr_210x * fac1 * wt;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += hrr_110x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += hrr_110x * fac1 * trr_10z;
                double hrr_010x = trr_10x - xjxi * 1;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout3 += hrr_010x * trr_20y * wt;
                gout4 += hrr_010x * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout5 += hrr_010x * fac1 * trr_20z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout6 += trr_20x * hrr_010y * wt;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout7 += trr_10x * hrr_110y * wt;
                gout8 += trr_10x * hrr_010y * trr_10z;
                double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                double hrr_210y = trr_30y - yjyi * trr_20y;
                gout9 += 1 * hrr_210y * wt;
                gout10 += 1 * hrr_110y * trr_10z;
                gout11 += 1 * hrr_010y * trr_20z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout12 += trr_20x * fac1 * hrr_010z;
                gout13 += trr_10x * trr_10y * hrr_010z;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout14 += trr_10x * fac1 * hrr_110z;
                gout15 += 1 * trr_20y * hrr_010z;
                gout16 += 1 * trr_10y * hrr_110z;
                double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                double hrr_210z = trr_30z - zjzi * trr_20z;
                gout17 += 1 * fac1 * hrr_210z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 18 * naux + ksh_in_block * 1;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[6*naux + 0] = gout6;
        eri_tensor[7*naux + 0] = gout7;
        eri_tensor[8*naux + 0] = gout8;
        eri_tensor[9*naux + 0] = gout9;
        eri_tensor[10*naux + 0] = gout10;
        eri_tensor[11*naux + 0] = gout11;
        eri_tensor[12*naux + 0] = gout12;
        eri_tensor[13*naux + 0] = gout13;
        eri_tensor[14*naux + 0] = gout14;
        eri_tensor[15*naux + 0] = gout15;
        eri_tensor[16*naux + 0] = gout16;
        eri_tensor[17*naux + 0] = gout17;
    }
}

__device__ static
void int3c2e_bdiv_220(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 3;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(3, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 6*nst_per_block;
                rys_roots(3, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                double hrr_310x = trr_40x - xjxi * trr_30x;
                double hrr_210x = trr_30x - xjxi * trr_20x;
                double hrr_220x = hrr_310x - xjxi * hrr_210x;
                gout0 += hrr_220x * fac1 * wt;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                double hrr_120x = hrr_210x - xjxi * hrr_110x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += hrr_120x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += hrr_120x * fac1 * trr_10z;
                double hrr_010x = trr_10x - xjxi * 1;
                double hrr_020x = hrr_110x - xjxi * hrr_010x;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout3 += hrr_020x * trr_20y * wt;
                gout4 += hrr_020x * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout5 += hrr_020x * fac1 * trr_20z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout6 += hrr_210x * hrr_010y * wt;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout7 += hrr_110x * hrr_110y * wt;
                gout8 += hrr_110x * hrr_010y * trr_10z;
                double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                double hrr_210y = trr_30y - yjyi * trr_20y;
                gout9 += hrr_010x * hrr_210y * wt;
                gout10 += hrr_010x * hrr_110y * trr_10z;
                gout11 += hrr_010x * hrr_010y * trr_20z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout12 += hrr_210x * fac1 * hrr_010z;
                gout13 += hrr_110x * trr_10y * hrr_010z;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout14 += hrr_110x * fac1 * hrr_110z;
                gout15 += hrr_010x * trr_20y * hrr_010z;
                gout16 += hrr_010x * trr_10y * hrr_110z;
                double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                double hrr_210z = trr_30z - zjzi * trr_20z;
                gout17 += hrr_010x * fac1 * hrr_210z;
                double hrr_020y = hrr_110y - yjyi * hrr_010y;
                gout18 += trr_20x * hrr_020y * wt;
                double hrr_120y = hrr_210y - yjyi * hrr_110y;
                gout19 += trr_10x * hrr_120y * wt;
                gout20 += trr_10x * hrr_020y * trr_10z;
                double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                double hrr_310y = trr_40y - yjyi * trr_30y;
                double hrr_220y = hrr_310y - yjyi * hrr_210y;
                gout21 += 1 * hrr_220y * wt;
                gout22 += 1 * hrr_120y * trr_10z;
                gout23 += 1 * hrr_020y * trr_20z;
                gout24 += trr_20x * hrr_010y * hrr_010z;
                gout25 += trr_10x * hrr_110y * hrr_010z;
                gout26 += trr_10x * hrr_010y * hrr_110z;
                gout27 += 1 * hrr_210y * hrr_010z;
                gout28 += 1 * hrr_110y * hrr_110z;
                gout29 += 1 * hrr_010y * hrr_210z;
                double hrr_020z = hrr_110z - zjzi * hrr_010z;
                gout30 += trr_20x * fac1 * hrr_020z;
                gout31 += trr_10x * trr_10y * hrr_020z;
                double hrr_120z = hrr_210z - zjzi * hrr_110z;
                gout32 += trr_10x * fac1 * hrr_120z;
                gout33 += 1 * trr_20y * hrr_020z;
                gout34 += 1 * trr_10y * hrr_120z;
                double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                double hrr_310z = trr_40z - zjzi * trr_30z;
                double hrr_220z = hrr_310z - zjzi * hrr_210z;
                gout35 += 1 * fac1 * hrr_220z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 36 * naux + ksh_in_block * 1;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[6*naux + 0] = gout6;
        eri_tensor[7*naux + 0] = gout7;
        eri_tensor[8*naux + 0] = gout8;
        eri_tensor[9*naux + 0] = gout9;
        eri_tensor[10*naux + 0] = gout10;
        eri_tensor[11*naux + 0] = gout11;
        eri_tensor[12*naux + 0] = gout12;
        eri_tensor[13*naux + 0] = gout13;
        eri_tensor[14*naux + 0] = gout14;
        eri_tensor[15*naux + 0] = gout15;
        eri_tensor[16*naux + 0] = gout16;
        eri_tensor[17*naux + 0] = gout17;
        eri_tensor[18*naux + 0] = gout18;
        eri_tensor[19*naux + 0] = gout19;
        eri_tensor[20*naux + 0] = gout20;
        eri_tensor[21*naux + 0] = gout21;
        eri_tensor[22*naux + 0] = gout22;
        eri_tensor[23*naux + 0] = gout23;
        eri_tensor[24*naux + 0] = gout24;
        eri_tensor[25*naux + 0] = gout25;
        eri_tensor[26*naux + 0] = gout26;
        eri_tensor[27*naux + 0] = gout27;
        eri_tensor[28*naux + 0] = gout28;
        eri_tensor[29*naux + 0] = gout29;
        eri_tensor[30*naux + 0] = gout30;
        eri_tensor[31*naux + 0] = gout31;
        eri_tensor[32*naux + 0] = gout32;
        eri_tensor[33*naux + 0] = gout33;
        eri_tensor[34*naux + 0] = gout34;
        eri_tensor[35*naux + 0] = gout35;
    }
}

__device__ static
void int3c2e_bdiv_001(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 1;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(1, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(1, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 2*nst_per_block;
                rys_roots(1, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(1, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double trr_01x = cpx * 1;
                gout0 += trr_01x * fac1 * wt;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout1 += 1 * trr_01y * wt;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout2 += 1 * fac1 * trr_01z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 1 * naux + ksh_in_block * 3;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout1;
        eri_tensor[0*naux + 2] = gout2;
    }
}

__device__ static
void int3c2e_bdiv_101(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                gout0 += trr_11x * fac1 * wt;
                double trr_01x = cpx * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += trr_01x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += trr_01x * fac1 * trr_10z;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout3 += trr_10x * trr_01y * wt;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout4 += 1 * trr_11y * wt;
                gout5 += 1 * trr_01y * trr_10z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout6 += trr_10x * fac1 * trr_01z;
                gout7 += 1 * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout8 += 1 * fac1 * trr_11z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 3 * naux + ksh_in_block * 3;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout3;
        eri_tensor[0*naux + 2] = gout6;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[1*naux + 1] = gout4;
        eri_tensor[1*naux + 2] = gout7;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[2*naux + 1] = gout5;
        eri_tensor[2*naux + 2] = gout8;
    }
}

__device__ static
void int3c2e_bdiv_111(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double hrr_111x = trr_21x - xjxi * trr_11x;
                gout0 += hrr_111x * fac1 * wt;
                double trr_01x = cpx * 1;
                double hrr_011x = trr_11x - xjxi * trr_01x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += hrr_011x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += hrr_011x * fac1 * trr_10z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout3 += trr_11x * hrr_010y * wt;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout4 += trr_01x * hrr_110y * wt;
                gout5 += trr_01x * hrr_010y * trr_10z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout6 += trr_11x * fac1 * hrr_010z;
                gout7 += trr_01x * trr_10y * hrr_010z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout8 += trr_01x * fac1 * hrr_110z;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout9 += hrr_110x * trr_01y * wt;
                double hrr_010x = trr_10x - xjxi * 1;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout10 += hrr_010x * trr_11y * wt;
                gout11 += hrr_010x * trr_01y * trr_10z;
                double hrr_011y = trr_11y - yjyi * trr_01y;
                gout12 += trr_10x * hrr_011y * wt;
                double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                double hrr_111y = trr_21y - yjyi * trr_11y;
                gout13 += 1 * hrr_111y * wt;
                gout14 += 1 * hrr_011y * trr_10z;
                gout15 += trr_10x * trr_01y * hrr_010z;
                gout16 += 1 * trr_11y * hrr_010z;
                gout17 += 1 * trr_01y * hrr_110z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout18 += hrr_110x * fac1 * trr_01z;
                gout19 += hrr_010x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout20 += hrr_010x * fac1 * trr_11z;
                gout21 += trr_10x * hrr_010y * trr_01z;
                gout22 += 1 * hrr_110y * trr_01z;
                gout23 += 1 * hrr_010y * trr_11z;
                double hrr_011z = trr_11z - zjzi * trr_01z;
                gout24 += trr_10x * fac1 * hrr_011z;
                gout25 += 1 * trr_10y * hrr_011z;
                double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                double hrr_111z = trr_21z - zjzi * trr_11z;
                gout26 += 1 * fac1 * hrr_111z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 9 * naux + ksh_in_block * 3;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout9;
        eri_tensor[0*naux + 2] = gout18;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[1*naux + 1] = gout10;
        eri_tensor[1*naux + 2] = gout19;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[2*naux + 1] = gout11;
        eri_tensor[2*naux + 2] = gout20;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[3*naux + 1] = gout12;
        eri_tensor[3*naux + 2] = gout21;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[4*naux + 1] = gout13;
        eri_tensor[4*naux + 2] = gout22;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[5*naux + 1] = gout14;
        eri_tensor[5*naux + 2] = gout23;
        eri_tensor[6*naux + 0] = gout6;
        eri_tensor[6*naux + 1] = gout15;
        eri_tensor[6*naux + 2] = gout24;
        eri_tensor[7*naux + 0] = gout7;
        eri_tensor[7*naux + 1] = gout16;
        eri_tensor[7*naux + 2] = gout25;
        eri_tensor[8*naux + 0] = gout8;
        eri_tensor[8*naux + 1] = gout17;
        eri_tensor[8*naux + 2] = gout26;
    }
}

__device__ static
void int3c2e_bdiv_201(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                gout0 += trr_21x * fac1 * wt;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += trr_11x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += trr_11x * fac1 * trr_10z;
                double trr_01x = cpx * 1;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout3 += trr_01x * trr_20y * wt;
                gout4 += trr_01x * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout5 += trr_01x * fac1 * trr_20z;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout6 += trr_20x * trr_01y * wt;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout7 += trr_10x * trr_11y * wt;
                gout8 += trr_10x * trr_01y * trr_10z;
                double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                gout9 += 1 * trr_21y * wt;
                gout10 += 1 * trr_11y * trr_10z;
                gout11 += 1 * trr_01y * trr_20z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout12 += trr_20x * fac1 * trr_01z;
                gout13 += trr_10x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout14 += trr_10x * fac1 * trr_11z;
                gout15 += 1 * trr_20y * trr_01z;
                gout16 += 1 * trr_10y * trr_11z;
                double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                gout17 += 1 * fac1 * trr_21z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 6 * naux + ksh_in_block * 3;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout6;
        eri_tensor[0*naux + 2] = gout12;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[1*naux + 1] = gout7;
        eri_tensor[1*naux + 2] = gout13;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[2*naux + 1] = gout8;
        eri_tensor[2*naux + 2] = gout14;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[3*naux + 1] = gout9;
        eri_tensor[3*naux + 2] = gout15;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[4*naux + 1] = gout10;
        eri_tensor[4*naux + 2] = gout16;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[5*naux + 1] = gout11;
        eri_tensor[5*naux + 2] = gout17;
    }
}

__device__ static
void int3c2e_bdiv_211(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 3;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        double gout36 = 0;
        double gout37 = 0;
        double gout38 = 0;
        double gout39 = 0;
        double gout40 = 0;
        double gout41 = 0;
        double gout42 = 0;
        double gout43 = 0;
        double gout44 = 0;
        double gout45 = 0;
        double gout46 = 0;
        double gout47 = 0;
        double gout48 = 0;
        double gout49 = 0;
        double gout50 = 0;
        double gout51 = 0;
        double gout52 = 0;
        double gout53 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(3, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 6*nst_per_block;
                rys_roots(3, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                double hrr_211x = trr_31x - xjxi * trr_21x;
                gout0 += hrr_211x * fac1 * wt;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double hrr_111x = trr_21x - xjxi * trr_11x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += hrr_111x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += hrr_111x * fac1 * trr_10z;
                double trr_01x = cpx * 1;
                double hrr_011x = trr_11x - xjxi * trr_01x;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout3 += hrr_011x * trr_20y * wt;
                gout4 += hrr_011x * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout5 += hrr_011x * fac1 * trr_20z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout6 += trr_21x * hrr_010y * wt;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout7 += trr_11x * hrr_110y * wt;
                gout8 += trr_11x * hrr_010y * trr_10z;
                double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                double hrr_210y = trr_30y - yjyi * trr_20y;
                gout9 += trr_01x * hrr_210y * wt;
                gout10 += trr_01x * hrr_110y * trr_10z;
                gout11 += trr_01x * hrr_010y * trr_20z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout12 += trr_21x * fac1 * hrr_010z;
                gout13 += trr_11x * trr_10y * hrr_010z;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout14 += trr_11x * fac1 * hrr_110z;
                gout15 += trr_01x * trr_20y * hrr_010z;
                gout16 += trr_01x * trr_10y * hrr_110z;
                double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                double hrr_210z = trr_30z - zjzi * trr_20z;
                gout17 += trr_01x * fac1 * hrr_210z;
                double hrr_210x = trr_30x - xjxi * trr_20x;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout18 += hrr_210x * trr_01y * wt;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout19 += hrr_110x * trr_11y * wt;
                gout20 += hrr_110x * trr_01y * trr_10z;
                double hrr_010x = trr_10x - xjxi * 1;
                double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                gout21 += hrr_010x * trr_21y * wt;
                gout22 += hrr_010x * trr_11y * trr_10z;
                gout23 += hrr_010x * trr_01y * trr_20z;
                double hrr_011y = trr_11y - yjyi * trr_01y;
                gout24 += trr_20x * hrr_011y * wt;
                double hrr_111y = trr_21y - yjyi * trr_11y;
                gout25 += trr_10x * hrr_111y * wt;
                gout26 += trr_10x * hrr_011y * trr_10z;
                double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                double hrr_211y = trr_31y - yjyi * trr_21y;
                gout27 += 1 * hrr_211y * wt;
                gout28 += 1 * hrr_111y * trr_10z;
                gout29 += 1 * hrr_011y * trr_20z;
                gout30 += trr_20x * trr_01y * hrr_010z;
                gout31 += trr_10x * trr_11y * hrr_010z;
                gout32 += trr_10x * trr_01y * hrr_110z;
                gout33 += 1 * trr_21y * hrr_010z;
                gout34 += 1 * trr_11y * hrr_110z;
                gout35 += 1 * trr_01y * hrr_210z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout36 += hrr_210x * fac1 * trr_01z;
                gout37 += hrr_110x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout38 += hrr_110x * fac1 * trr_11z;
                gout39 += hrr_010x * trr_20y * trr_01z;
                gout40 += hrr_010x * trr_10y * trr_11z;
                double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                gout41 += hrr_010x * fac1 * trr_21z;
                gout42 += trr_20x * hrr_010y * trr_01z;
                gout43 += trr_10x * hrr_110y * trr_01z;
                gout44 += trr_10x * hrr_010y * trr_11z;
                gout45 += 1 * hrr_210y * trr_01z;
                gout46 += 1 * hrr_110y * trr_11z;
                gout47 += 1 * hrr_010y * trr_21z;
                double hrr_011z = trr_11z - zjzi * trr_01z;
                gout48 += trr_20x * fac1 * hrr_011z;
                gout49 += trr_10x * trr_10y * hrr_011z;
                double hrr_111z = trr_21z - zjzi * trr_11z;
                gout50 += trr_10x * fac1 * hrr_111z;
                gout51 += 1 * trr_20y * hrr_011z;
                gout52 += 1 * trr_10y * hrr_111z;
                double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                double hrr_211z = trr_31z - zjzi * trr_21z;
                gout53 += 1 * fac1 * hrr_211z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 18 * naux + ksh_in_block * 3;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout18;
        eri_tensor[0*naux + 2] = gout36;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[1*naux + 1] = gout19;
        eri_tensor[1*naux + 2] = gout37;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[2*naux + 1] = gout20;
        eri_tensor[2*naux + 2] = gout38;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[3*naux + 1] = gout21;
        eri_tensor[3*naux + 2] = gout39;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[4*naux + 1] = gout22;
        eri_tensor[4*naux + 2] = gout40;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[5*naux + 1] = gout23;
        eri_tensor[5*naux + 2] = gout41;
        eri_tensor[6*naux + 0] = gout6;
        eri_tensor[6*naux + 1] = gout24;
        eri_tensor[6*naux + 2] = gout42;
        eri_tensor[7*naux + 0] = gout7;
        eri_tensor[7*naux + 1] = gout25;
        eri_tensor[7*naux + 2] = gout43;
        eri_tensor[8*naux + 0] = gout8;
        eri_tensor[8*naux + 1] = gout26;
        eri_tensor[8*naux + 2] = gout44;
        eri_tensor[9*naux + 0] = gout9;
        eri_tensor[9*naux + 1] = gout27;
        eri_tensor[9*naux + 2] = gout45;
        eri_tensor[10*naux + 0] = gout10;
        eri_tensor[10*naux + 1] = gout28;
        eri_tensor[10*naux + 2] = gout46;
        eri_tensor[11*naux + 0] = gout11;
        eri_tensor[11*naux + 1] = gout29;
        eri_tensor[11*naux + 2] = gout47;
        eri_tensor[12*naux + 0] = gout12;
        eri_tensor[12*naux + 1] = gout30;
        eri_tensor[12*naux + 2] = gout48;
        eri_tensor[13*naux + 0] = gout13;
        eri_tensor[13*naux + 1] = gout31;
        eri_tensor[13*naux + 2] = gout49;
        eri_tensor[14*naux + 0] = gout14;
        eri_tensor[14*naux + 1] = gout32;
        eri_tensor[14*naux + 2] = gout50;
        eri_tensor[15*naux + 0] = gout15;
        eri_tensor[15*naux + 1] = gout33;
        eri_tensor[15*naux + 2] = gout51;
        eri_tensor[16*naux + 0] = gout16;
        eri_tensor[16*naux + 1] = gout34;
        eri_tensor[16*naux + 2] = gout52;
        eri_tensor[17*naux + 0] = gout17;
        eri_tensor[17*naux + 1] = gout35;
        eri_tensor[17*naux + 2] = gout53;
    }
}

__device__
void int3c2e_bdiv_221(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int thread_id = threadIdx.x;
    int st_id = thread_id % 64;
    int gout_id = thread_id / 64;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 3;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 128;
    double *gy = gx + 1152;
    double *gz = gy + 1152;
    double *Rpq = gz + 1152;
    double *rjri = Rpq + 192;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    if (gout_id == 0) {
        gx[0] = 1.;
    }

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 64) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[64] = yjyi;
            rjri[128] = zjzi;
            rjri[192] = rr_ij;
        }
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double s0, s1, s2;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[64] * aj_aij + ri[1];
            double zij = rjri[128] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[192];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(3, theta_rr, rw, 64, gout_id, 4);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                __syncthreads();
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = gout_id; irys < 3; irys+=4) {
                    rw[ irys*2   *64] *= theta_fac;
                    rw[(irys*2+1)*64] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 384;
                rys_roots(3, theta_rr, rw1, 64, gout_id, 4);
                rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                __syncthreads();
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = gout_id; irys < 3; irys+=4) {
                    rw[ irys*2   *64] *= theta_fac;
                    rw[(irys*2+1)*64] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*128];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                for (int n = gout_id; n < 3; n += 4) {
                    if (n == 2) {
                        gz[0] = rw[irys*128+64];
                    }
                    double *_gx = gx + n * 1152;
                    double xjxi = rjri[n * 64];
                    double Rpa = xjxi * aj_aij;
                    double c0x = Rpa - rt_aij * Rpq[n * 64];
                    s0 = _gx[0];
                    s1 = c0x * s0;
                    _gx[64] = s1;
                    s2 = c0x * s1 + 1 * b10 * s0;
                    _gx[128] = s2;
                    s0 = s1;
                    s1 = s2;
                    s2 = c0x * s1 + 2 * b10 * s0;
                    _gx[192] = s2;
                    s0 = s1;
                    s1 = s2;
                    s2 = c0x * s1 + 3 * b10 * s0;
                    _gx[256] = s2;
                    double cpx = rt_ak * Rpq[n * 64];
                    s0 = _gx[0];
                    s1 = cpx * s0;
                    _gx[576] = s1;
                    s0 = _gx[64];
                    s1 = cpx * s0;
                    s1 += 1 * b00 * _gx[0];
                    _gx[640] = s1;
                    s0 = _gx[128];
                    s1 = cpx * s0;
                    s1 += 2 * b00 * _gx[64];
                    _gx[704] = s1;
                    s0 = _gx[192];
                    s1 = cpx * s0;
                    s1 += 3 * b00 * _gx[128];
                    _gx[768] = s1;
                    s0 = _gx[256];
                    s1 = cpx * s0;
                    s1 += 4 * b00 * _gx[192];
                    _gx[832] = s1;
                    s1 = _gx[256];
                    s0 = _gx[192];
                    _gx[384] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[128];
                    _gx[320] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[64];
                    _gx[256] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[0];
                    _gx[192] = s1 - xjxi * s0;
                    s1 = _gx[384];
                    s0 = _gx[320];
                    _gx[512] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[256];
                    _gx[448] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[192];
                    _gx[384] = s1 - xjxi * s0;
                    s1 = _gx[832];
                    s0 = _gx[768];
                    _gx[960] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[704];
                    _gx[896] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[640];
                    _gx[832] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[576];
                    _gx[768] = s1 - xjxi * s0;
                    s1 = _gx[960];
                    s0 = _gx[896];
                    _gx[1088] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[832];
                    _gx[1024] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[768];
                    _gx[960] = s1 - xjxi * s0;
                }
                __syncthreads();
                switch (gout_id) {
                case 0:
                gout0 += gx[1088] * gy[0] * gz[0];
                gout1 += gx[448] * gy[640] * gz[0];
                gout2 += gx[448] * gy[0] * gz[640];
                gout3 += gx[960] * gy[64] * gz[64];
                gout4 += gx[384] * gy[576] * gz[128];
                gout5 += gx[320] * gy[192] * gz[576];
                gout6 += gx[832] * gy[192] * gz[64];
                gout7 += gx[192] * gy[896] * gz[0];
                gout8 += gx[192] * gy[256] * gz[640];
                gout9 += gx[896] * gy[0] * gz[192];
                gout10 += gx[256] * gy[640] * gz[192];
                gout11 += gx[256] * gy[0] * gz[832];
                gout12 += gx[768] * gy[64] * gz[256];
                gout13 += gx[192] * gy[576] * gz[320];
                gout14 += gx[128] * gy[384] * gz[576];
                gout15 += gx[640] * gy[384] * gz[64];
                gout16 += gx[0] * gy[1088] * gz[0];
                gout17 += gx[0] * gy[448] * gz[640];
                gout18 += gx[704] * gy[192] * gz[192];
                gout19 += gx[64] * gy[832] * gz[192];
                gout20 += gx[64] * gy[192] * gz[832];
                gout21 += gx[576] * gy[256] * gz[256];
                gout22 += gx[0] * gy[768] * gz[320];
                gout23 += gx[128] * gy[0] * gz[960];
                gout24 += gx[640] * gy[0] * gz[448];
                gout25 += gx[0] * gy[704] * gz[384];
                gout26 += gx[0] * gy[64] * gz[1024];
                break;
                case 1:
                gout0 += gx[512] * gy[576] * gz[0];
                gout1 += gx[448] * gy[64] * gz[576];
                gout2 += gx[960] * gy[128] * gz[0];
                gout3 += gx[384] * gy[640] * gz[64];
                gout4 += gx[384] * gy[0] * gz[704];
                gout5 += gx[832] * gy[256] * gz[0];
                gout6 += gx[256] * gy[768] * gz[64];
                gout7 += gx[192] * gy[320] * gz[576];
                gout8 += gx[768] * gy[192] * gz[128];
                gout9 += gx[320] * gy[576] * gz[192];
                gout10 += gx[256] * gy[64] * gz[768];
                gout11 += gx[768] * gy[128] * gz[192];
                gout12 += gx[192] * gy[640] * gz[256];
                gout13 += gx[192] * gy[0] * gz[896];
                gout14 += gx[640] * gy[448] * gz[0];
                gout15 += gx[64] * gy[960] * gz[64];
                gout16 += gx[0] * gy[512] * gz[576];
                gout17 += gx[576] * gy[384] * gz[128];
                gout18 += gx[128] * gy[768] * gz[192];
                gout19 += gx[64] * gy[256] * gz[768];
                gout20 += gx[576] * gy[320] * gz[192];
                gout21 += gx[0] * gy[832] * gz[256];
                gout22 += gx[0] * gy[192] * gz[896];
                gout23 += gx[640] * gy[64] * gz[384];
                gout24 += gx[64] * gy[576] * gz[448];
                gout25 += gx[0] * gy[128] * gz[960];
                gout26 += gx[576] * gy[0] * gz[512];
                break;
                case 2:
                gout0 += gx[512] * gy[0] * gz[576];
                gout1 += gx[1024] * gy[0] * gz[64];
                gout2 += gx[384] * gy[704] * gz[0];
                gout3 += gx[384] * gy[64] * gz[640];
                gout4 += gx[896] * gy[192] * gz[0];
                gout5 += gx[256] * gy[832] * gz[0];
                gout6 += gx[256] * gy[192] * gz[640];
                gout7 += gx[768] * gy[256] * gz[64];
                gout8 += gx[192] * gy[768] * gz[128];
                gout9 += gx[320] * gy[0] * gz[768];
                gout10 += gx[832] * gy[0] * gz[256];
                gout11 += gx[192] * gy[704] * gz[192];
                gout12 += gx[192] * gy[64] * gz[832];
                gout13 += gx[704] * gy[384] * gz[0];
                gout14 += gx[64] * gy[1024] * gz[0];
                gout15 += gx[64] * gy[384] * gz[640];
                gout16 += gx[576] * gy[448] * gz[64];
                gout17 += gx[0] * gy[960] * gz[128];
                gout18 += gx[128] * gy[192] * gz[768];
                gout19 += gx[640] * gy[192] * gz[256];
                gout20 += gx[0] * gy[896] * gz[192];
                gout21 += gx[0] * gy[256] * gz[832];
                gout22 += gx[704] * gy[0] * gz[384];
                gout23 += gx[64] * gy[640] * gz[384];
                gout24 += gx[64] * gy[0] * gz[1024];
                gout25 += gx[576] * gy[64] * gz[448];
                gout26 += gx[0] * gy[576] * gz[512];
                break;
                case 3:
                gout0 += gx[1024] * gy[64] * gz[0];
                gout1 += gx[448] * gy[576] * gz[64];
                gout2 += gx[384] * gy[128] * gz[576];
                gout3 += gx[960] * gy[0] * gz[128];
                gout4 += gx[320] * gy[768] * gz[0];
                gout5 += gx[256] * gy[256] * gz[576];
                gout6 += gx[768] * gy[320] * gz[0];
                gout7 += gx[192] * gy[832] * gz[64];
                gout8 += gx[192] * gy[192] * gz[704];
                gout9 += gx[832] * gy[64] * gz[192];
                gout10 += gx[256] * gy[576] * gz[256];
                gout11 += gx[192] * gy[128] * gz[768];
                gout12 += gx[768] * gy[0] * gz[320];
                gout13 += gx[128] * gy[960] * gz[0];
                gout14 += gx[64] * gy[448] * gz[576];
                gout15 += gx[576] * gy[512] * gz[0];
                gout16 += gx[0] * gy[1024] * gz[64];
                gout17 += gx[0] * gy[384] * gz[704];
                gout18 += gx[640] * gy[256] * gz[192];
                gout19 += gx[64] * gy[768] * gz[256];
                gout20 += gx[0] * gy[320] * gz[768];
                gout21 += gx[576] * gy[192] * gz[320];
                gout22 += gx[128] * gy[576] * gz[384];
                gout23 += gx[64] * gy[64] * gz[960];
                gout24 += gx[576] * gy[128] * gz[384];
                gout25 += gx[0] * gy[640] * gz[448];
                gout26 += gx[0] * gy[0] * gz[1088];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            int *ao_loc = envs.ao_loc;
            int k0 = ao_loc[ksh0] - ao_loc[nbas];
            double *eri_tensor = out_local + shl_pair_in_block * 36 * naux + k0 + ksh_in_block * 3;
            switch (gout_id) {
            case 0:
            eri_tensor[0*naux + 0] = gout0;
            eri_tensor[1*naux + 1] = gout1;
            eri_tensor[2*naux + 2] = gout2;
            eri_tensor[4*naux + 0] = gout3;
            eri_tensor[5*naux + 1] = gout4;
            eri_tensor[6*naux + 2] = gout5;
            eri_tensor[8*naux + 0] = gout6;
            eri_tensor[9*naux + 1] = gout7;
            eri_tensor[10*naux + 2] = gout8;
            eri_tensor[12*naux + 0] = gout9;
            eri_tensor[13*naux + 1] = gout10;
            eri_tensor[14*naux + 2] = gout11;
            eri_tensor[16*naux + 0] = gout12;
            eri_tensor[17*naux + 1] = gout13;
            eri_tensor[18*naux + 2] = gout14;
            eri_tensor[20*naux + 0] = gout15;
            eri_tensor[21*naux + 1] = gout16;
            eri_tensor[22*naux + 2] = gout17;
            eri_tensor[24*naux + 0] = gout18;
            eri_tensor[25*naux + 1] = gout19;
            eri_tensor[26*naux + 2] = gout20;
            eri_tensor[28*naux + 0] = gout21;
            eri_tensor[29*naux + 1] = gout22;
            eri_tensor[30*naux + 2] = gout23;
            eri_tensor[32*naux + 0] = gout24;
            eri_tensor[33*naux + 1] = gout25;
            eri_tensor[34*naux + 2] = gout26;
            break;
            case 1:
            eri_tensor[0*naux + 1] = gout0;
            eri_tensor[1*naux + 2] = gout1;
            eri_tensor[3*naux + 0] = gout2;
            eri_tensor[4*naux + 1] = gout3;
            eri_tensor[5*naux + 2] = gout4;
            eri_tensor[7*naux + 0] = gout5;
            eri_tensor[8*naux + 1] = gout6;
            eri_tensor[9*naux + 2] = gout7;
            eri_tensor[11*naux + 0] = gout8;
            eri_tensor[12*naux + 1] = gout9;
            eri_tensor[13*naux + 2] = gout10;
            eri_tensor[15*naux + 0] = gout11;
            eri_tensor[16*naux + 1] = gout12;
            eri_tensor[17*naux + 2] = gout13;
            eri_tensor[19*naux + 0] = gout14;
            eri_tensor[20*naux + 1] = gout15;
            eri_tensor[21*naux + 2] = gout16;
            eri_tensor[23*naux + 0] = gout17;
            eri_tensor[24*naux + 1] = gout18;
            eri_tensor[25*naux + 2] = gout19;
            eri_tensor[27*naux + 0] = gout20;
            eri_tensor[28*naux + 1] = gout21;
            eri_tensor[29*naux + 2] = gout22;
            eri_tensor[31*naux + 0] = gout23;
            eri_tensor[32*naux + 1] = gout24;
            eri_tensor[33*naux + 2] = gout25;
            eri_tensor[35*naux + 0] = gout26;
            break;
            case 2:
            eri_tensor[0*naux + 2] = gout0;
            eri_tensor[2*naux + 0] = gout1;
            eri_tensor[3*naux + 1] = gout2;
            eri_tensor[4*naux + 2] = gout3;
            eri_tensor[6*naux + 0] = gout4;
            eri_tensor[7*naux + 1] = gout5;
            eri_tensor[8*naux + 2] = gout6;
            eri_tensor[10*naux + 0] = gout7;
            eri_tensor[11*naux + 1] = gout8;
            eri_tensor[12*naux + 2] = gout9;
            eri_tensor[14*naux + 0] = gout10;
            eri_tensor[15*naux + 1] = gout11;
            eri_tensor[16*naux + 2] = gout12;
            eri_tensor[18*naux + 0] = gout13;
            eri_tensor[19*naux + 1] = gout14;
            eri_tensor[20*naux + 2] = gout15;
            eri_tensor[22*naux + 0] = gout16;
            eri_tensor[23*naux + 1] = gout17;
            eri_tensor[24*naux + 2] = gout18;
            eri_tensor[26*naux + 0] = gout19;
            eri_tensor[27*naux + 1] = gout20;
            eri_tensor[28*naux + 2] = gout21;
            eri_tensor[30*naux + 0] = gout22;
            eri_tensor[31*naux + 1] = gout23;
            eri_tensor[32*naux + 2] = gout24;
            eri_tensor[34*naux + 0] = gout25;
            eri_tensor[35*naux + 1] = gout26;
            break;
            case 3:
            eri_tensor[1*naux + 0] = gout0;
            eri_tensor[2*naux + 1] = gout1;
            eri_tensor[3*naux + 2] = gout2;
            eri_tensor[5*naux + 0] = gout3;
            eri_tensor[6*naux + 1] = gout4;
            eri_tensor[7*naux + 2] = gout5;
            eri_tensor[9*naux + 0] = gout6;
            eri_tensor[10*naux + 1] = gout7;
            eri_tensor[11*naux + 2] = gout8;
            eri_tensor[13*naux + 0] = gout9;
            eri_tensor[14*naux + 1] = gout10;
            eri_tensor[15*naux + 2] = gout11;
            eri_tensor[17*naux + 0] = gout12;
            eri_tensor[18*naux + 1] = gout13;
            eri_tensor[19*naux + 2] = gout14;
            eri_tensor[21*naux + 0] = gout15;
            eri_tensor[22*naux + 1] = gout16;
            eri_tensor[23*naux + 2] = gout17;
            eri_tensor[25*naux + 0] = gout18;
            eri_tensor[26*naux + 1] = gout19;
            eri_tensor[27*naux + 2] = gout20;
            eri_tensor[29*naux + 0] = gout21;
            eri_tensor[30*naux + 1] = gout22;
            eri_tensor[31*naux + 2] = gout23;
            eri_tensor[33*naux + 0] = gout24;
            eri_tensor[34*naux + 1] = gout25;
            eri_tensor[35*naux + 2] = gout26;
            break;
            }
        }
    }
}

__device__ static
void int3c2e_bdiv_002(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_ak = rt_aa * aij;
                double b01 = .5/ak * (1 - rt_ak);
                double cpx = xpq*rt_ak;
                double trr_01x = cpx * 1;
                double trr_02x = cpx * trr_01x + 1*b01 * 1;
                gout0 += trr_02x * fac1 * wt;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout1 += trr_01x * trr_01y * wt;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout2 += trr_01x * fac1 * trr_01z;
                double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                gout3 += 1 * trr_02y * wt;
                gout4 += 1 * trr_01y * trr_01z;
                double trr_02z = cpz * trr_01z + 1*b01 * wt;
                gout5 += 1 * fac1 * trr_02z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 1 * naux + ksh_in_block * 6;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout1;
        eri_tensor[0*naux + 2] = gout2;
        eri_tensor[0*naux + 3] = gout3;
        eri_tensor[0*naux + 4] = gout4;
        eri_tensor[0*naux + 5] = gout5;
    }
}

__device__ static
void int3c2e_bdiv_102(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 2;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double b01 = .5/ak * (1 - rt_ak);
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double trr_01x = cpx * 1;
                double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                gout0 += trr_12x * fac1 * wt;
                double trr_02x = cpx * trr_01x + 1*b01 * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += trr_02x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += trr_02x * fac1 * trr_10z;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout3 += trr_11x * trr_01y * wt;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout4 += trr_01x * trr_11y * wt;
                gout5 += trr_01x * trr_01y * trr_10z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout6 += trr_11x * fac1 * trr_01z;
                gout7 += trr_01x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout8 += trr_01x * fac1 * trr_11z;
                double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                gout9 += trr_10x * trr_02y * wt;
                double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                gout10 += 1 * trr_12y * wt;
                gout11 += 1 * trr_02y * trr_10z;
                gout12 += trr_10x * trr_01y * trr_01z;
                gout13 += 1 * trr_11y * trr_01z;
                gout14 += 1 * trr_01y * trr_11z;
                double trr_02z = cpz * trr_01z + 1*b01 * wt;
                gout15 += trr_10x * fac1 * trr_02z;
                gout16 += 1 * trr_10y * trr_02z;
                double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                gout17 += 1 * fac1 * trr_12z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 3 * naux + ksh_in_block * 6;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout3;
        eri_tensor[0*naux + 2] = gout6;
        eri_tensor[0*naux + 3] = gout9;
        eri_tensor[0*naux + 4] = gout12;
        eri_tensor[0*naux + 5] = gout15;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[1*naux + 1] = gout4;
        eri_tensor[1*naux + 2] = gout7;
        eri_tensor[1*naux + 3] = gout10;
        eri_tensor[1*naux + 4] = gout13;
        eri_tensor[1*naux + 5] = gout16;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[2*naux + 1] = gout5;
        eri_tensor[2*naux + 2] = gout8;
        eri_tensor[2*naux + 3] = gout11;
        eri_tensor[2*naux + 4] = gout14;
        eri_tensor[2*naux + 5] = gout17;
    }
}

__device__ static
void int3c2e_bdiv_112(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 3;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        double gout36 = 0;
        double gout37 = 0;
        double gout38 = 0;
        double gout39 = 0;
        double gout40 = 0;
        double gout41 = 0;
        double gout42 = 0;
        double gout43 = 0;
        double gout44 = 0;
        double gout45 = 0;
        double gout46 = 0;
        double gout47 = 0;
        double gout48 = 0;
        double gout49 = 0;
        double gout50 = 0;
        double gout51 = 0;
        double gout52 = 0;
        double gout53 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(3, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 6*nst_per_block;
                rys_roots(3, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double b01 = .5/ak * (1 - rt_ak);
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                double trr_01x = cpx * 1;
                double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                double hrr_112x = trr_22x - xjxi * trr_12x;
                gout0 += hrr_112x * fac1 * wt;
                double trr_02x = cpx * trr_01x + 1*b01 * 1;
                double hrr_012x = trr_12x - xjxi * trr_02x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += hrr_012x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += hrr_012x * fac1 * trr_10z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout3 += trr_12x * hrr_010y * wt;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout4 += trr_02x * hrr_110y * wt;
                gout5 += trr_02x * hrr_010y * trr_10z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout6 += trr_12x * fac1 * hrr_010z;
                gout7 += trr_02x * trr_10y * hrr_010z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout8 += trr_02x * fac1 * hrr_110z;
                double hrr_111x = trr_21x - xjxi * trr_11x;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout9 += hrr_111x * trr_01y * wt;
                double hrr_011x = trr_11x - xjxi * trr_01x;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout10 += hrr_011x * trr_11y * wt;
                gout11 += hrr_011x * trr_01y * trr_10z;
                double hrr_011y = trr_11y - yjyi * trr_01y;
                gout12 += trr_11x * hrr_011y * wt;
                double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                double hrr_111y = trr_21y - yjyi * trr_11y;
                gout13 += trr_01x * hrr_111y * wt;
                gout14 += trr_01x * hrr_011y * trr_10z;
                gout15 += trr_11x * trr_01y * hrr_010z;
                gout16 += trr_01x * trr_11y * hrr_010z;
                gout17 += trr_01x * trr_01y * hrr_110z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout18 += hrr_111x * fac1 * trr_01z;
                gout19 += hrr_011x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout20 += hrr_011x * fac1 * trr_11z;
                gout21 += trr_11x * hrr_010y * trr_01z;
                gout22 += trr_01x * hrr_110y * trr_01z;
                gout23 += trr_01x * hrr_010y * trr_11z;
                double hrr_011z = trr_11z - zjzi * trr_01z;
                gout24 += trr_11x * fac1 * hrr_011z;
                gout25 += trr_01x * trr_10y * hrr_011z;
                double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                double hrr_111z = trr_21z - zjzi * trr_11z;
                gout26 += trr_01x * fac1 * hrr_111z;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                gout27 += hrr_110x * trr_02y * wt;
                double hrr_010x = trr_10x - xjxi * 1;
                double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                gout28 += hrr_010x * trr_12y * wt;
                gout29 += hrr_010x * trr_02y * trr_10z;
                double hrr_012y = trr_12y - yjyi * trr_02y;
                gout30 += trr_10x * hrr_012y * wt;
                double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                double hrr_112y = trr_22y - yjyi * trr_12y;
                gout31 += 1 * hrr_112y * wt;
                gout32 += 1 * hrr_012y * trr_10z;
                gout33 += trr_10x * trr_02y * hrr_010z;
                gout34 += 1 * trr_12y * hrr_010z;
                gout35 += 1 * trr_02y * hrr_110z;
                gout36 += hrr_110x * trr_01y * trr_01z;
                gout37 += hrr_010x * trr_11y * trr_01z;
                gout38 += hrr_010x * trr_01y * trr_11z;
                gout39 += trr_10x * hrr_011y * trr_01z;
                gout40 += 1 * hrr_111y * trr_01z;
                gout41 += 1 * hrr_011y * trr_11z;
                gout42 += trr_10x * trr_01y * hrr_011z;
                gout43 += 1 * trr_11y * hrr_011z;
                gout44 += 1 * trr_01y * hrr_111z;
                double trr_02z = cpz * trr_01z + 1*b01 * wt;
                gout45 += hrr_110x * fac1 * trr_02z;
                gout46 += hrr_010x * trr_10y * trr_02z;
                double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                gout47 += hrr_010x * fac1 * trr_12z;
                gout48 += trr_10x * hrr_010y * trr_02z;
                gout49 += 1 * hrr_110y * trr_02z;
                gout50 += 1 * hrr_010y * trr_12z;
                double hrr_012z = trr_12z - zjzi * trr_02z;
                gout51 += trr_10x * fac1 * hrr_012z;
                gout52 += 1 * trr_10y * hrr_012z;
                double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                double hrr_112z = trr_22z - zjzi * trr_12z;
                gout53 += 1 * fac1 * hrr_112z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 9 * naux + ksh_in_block * 6;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout9;
        eri_tensor[0*naux + 2] = gout18;
        eri_tensor[0*naux + 3] = gout27;
        eri_tensor[0*naux + 4] = gout36;
        eri_tensor[0*naux + 5] = gout45;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[1*naux + 1] = gout10;
        eri_tensor[1*naux + 2] = gout19;
        eri_tensor[1*naux + 3] = gout28;
        eri_tensor[1*naux + 4] = gout37;
        eri_tensor[1*naux + 5] = gout46;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[2*naux + 1] = gout11;
        eri_tensor[2*naux + 2] = gout20;
        eri_tensor[2*naux + 3] = gout29;
        eri_tensor[2*naux + 4] = gout38;
        eri_tensor[2*naux + 5] = gout47;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[3*naux + 1] = gout12;
        eri_tensor[3*naux + 2] = gout21;
        eri_tensor[3*naux + 3] = gout30;
        eri_tensor[3*naux + 4] = gout39;
        eri_tensor[3*naux + 5] = gout48;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[4*naux + 1] = gout13;
        eri_tensor[4*naux + 2] = gout22;
        eri_tensor[4*naux + 3] = gout31;
        eri_tensor[4*naux + 4] = gout40;
        eri_tensor[4*naux + 5] = gout49;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[5*naux + 1] = gout14;
        eri_tensor[5*naux + 2] = gout23;
        eri_tensor[5*naux + 3] = gout32;
        eri_tensor[5*naux + 4] = gout41;
        eri_tensor[5*naux + 5] = gout50;
        eri_tensor[6*naux + 0] = gout6;
        eri_tensor[6*naux + 1] = gout15;
        eri_tensor[6*naux + 2] = gout24;
        eri_tensor[6*naux + 3] = gout33;
        eri_tensor[6*naux + 4] = gout42;
        eri_tensor[6*naux + 5] = gout51;
        eri_tensor[7*naux + 0] = gout7;
        eri_tensor[7*naux + 1] = gout16;
        eri_tensor[7*naux + 2] = gout25;
        eri_tensor[7*naux + 3] = gout34;
        eri_tensor[7*naux + 4] = gout43;
        eri_tensor[7*naux + 5] = gout52;
        eri_tensor[8*naux + 0] = gout8;
        eri_tensor[8*naux + 1] = gout17;
        eri_tensor[8*naux + 2] = gout26;
        eri_tensor[8*naux + 3] = gout35;
        eri_tensor[8*naux + 4] = gout44;
        eri_tensor[8*naux + 5] = gout53;
    }
}

__device__ static
void int3c2e_bdiv_202(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    // For better load balance, consume blocks in the reversed order
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int nst_per_block = blockDim.x;
    int st_id = threadIdx.x;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 3;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw + nst_per_block * nroots*2;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(3, theta_rr, rw, nst_per_block, 0, 1);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 6*nst_per_block;
                rys_roots(3, theta_rr, rw1, nst_per_block, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *nst_per_block] *= theta_fac;
                    rw[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double b01 = .5/ak * (1 - rt_ak);
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                gout0 += trr_22x * fac1 * wt;
                double trr_01x = cpx * 1;
                double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout1 += trr_12x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout2 += trr_12x * fac1 * trr_10z;
                double trr_02x = cpx * trr_01x + 1*b01 * 1;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout3 += trr_02x * trr_20y * wt;
                gout4 += trr_02x * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout5 += trr_02x * fac1 * trr_20z;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout6 += trr_21x * trr_01y * wt;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout7 += trr_11x * trr_11y * wt;
                gout8 += trr_11x * trr_01y * trr_10z;
                double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                gout9 += trr_01x * trr_21y * wt;
                gout10 += trr_01x * trr_11y * trr_10z;
                gout11 += trr_01x * trr_01y * trr_20z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout12 += trr_21x * fac1 * trr_01z;
                gout13 += trr_11x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout14 += trr_11x * fac1 * trr_11z;
                gout15 += trr_01x * trr_20y * trr_01z;
                gout16 += trr_01x * trr_10y * trr_11z;
                double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                gout17 += trr_01x * fac1 * trr_21z;
                double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                gout18 += trr_20x * trr_02y * wt;
                double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                gout19 += trr_10x * trr_12y * wt;
                gout20 += trr_10x * trr_02y * trr_10z;
                double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                gout21 += 1 * trr_22y * wt;
                gout22 += 1 * trr_12y * trr_10z;
                gout23 += 1 * trr_02y * trr_20z;
                gout24 += trr_20x * trr_01y * trr_01z;
                gout25 += trr_10x * trr_11y * trr_01z;
                gout26 += trr_10x * trr_01y * trr_11z;
                gout27 += 1 * trr_21y * trr_01z;
                gout28 += 1 * trr_11y * trr_11z;
                gout29 += 1 * trr_01y * trr_21z;
                double trr_02z = cpz * trr_01z + 1*b01 * wt;
                gout30 += trr_20x * fac1 * trr_02z;
                gout31 += trr_10x * trr_10y * trr_02z;
                double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                gout32 += trr_10x * fac1 * trr_12z;
                gout33 += 1 * trr_20y * trr_02z;
                gout34 += 1 * trr_10y * trr_12z;
                double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                gout35 += 1 * fac1 * trr_22z;
            }
        }
        int *ao_loc = envs.ao_loc;
        int k0 = ao_loc[ksh0] - ao_loc[nbas];
        double *eri_tensor = out_local + k0 + shl_pair_in_block * 6 * naux + ksh_in_block * 6;
        eri_tensor[0*naux + 0] = gout0;
        eri_tensor[0*naux + 1] = gout6;
        eri_tensor[0*naux + 2] = gout12;
        eri_tensor[0*naux + 3] = gout18;
        eri_tensor[0*naux + 4] = gout24;
        eri_tensor[0*naux + 5] = gout30;
        eri_tensor[1*naux + 0] = gout1;
        eri_tensor[1*naux + 1] = gout7;
        eri_tensor[1*naux + 2] = gout13;
        eri_tensor[1*naux + 3] = gout19;
        eri_tensor[1*naux + 4] = gout25;
        eri_tensor[1*naux + 5] = gout31;
        eri_tensor[2*naux + 0] = gout2;
        eri_tensor[2*naux + 1] = gout8;
        eri_tensor[2*naux + 2] = gout14;
        eri_tensor[2*naux + 3] = gout20;
        eri_tensor[2*naux + 4] = gout26;
        eri_tensor[2*naux + 5] = gout32;
        eri_tensor[3*naux + 0] = gout3;
        eri_tensor[3*naux + 1] = gout9;
        eri_tensor[3*naux + 2] = gout15;
        eri_tensor[3*naux + 3] = gout21;
        eri_tensor[3*naux + 4] = gout27;
        eri_tensor[3*naux + 5] = gout33;
        eri_tensor[4*naux + 0] = gout4;
        eri_tensor[4*naux + 1] = gout10;
        eri_tensor[4*naux + 2] = gout16;
        eri_tensor[4*naux + 3] = gout22;
        eri_tensor[4*naux + 4] = gout28;
        eri_tensor[4*naux + 5] = gout34;
        eri_tensor[5*naux + 0] = gout5;
        eri_tensor[5*naux + 1] = gout11;
        eri_tensor[5*naux + 2] = gout17;
        eri_tensor[5*naux + 3] = gout23;
        eri_tensor[5*naux + 4] = gout29;
        eri_tensor[5*naux + 5] = gout35;
    }
}

__device__
void int3c2e_bdiv_212(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int shl_pair1 = bounds.shl_pair_offsets[sp_block_id+1];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int ksh1 = bounds.ksh_offsets[ksh_block_id+1];
    int nksh = ksh1 - ksh0;
    int nshl_pair = shl_pair1 - shl_pair0;
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int thread_id = threadIdx.x;
    int st_id = thread_id % 64;
    int gout_id = thread_id / 64;
    int *bas = envs.bas;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int ijkprim = ijprim * kprim;
    int nroots = 3;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 128;
    double *gy = gx + 1152;
    double *gz = gy + 1152;
    double *Rpq = gz + 1152;
    double *rjri = Rpq + 192;
    int naux = bounds.naux;
    double *out_local = out + bounds.ao_pair_loc[sp_block_id] * naux;

    if (gout_id == 0) {
        gx[0] = 1.;
    }

    int nst = nshl_pair * nksh;
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 64) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx % nksh;
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bounds.bas_ij_idx[shl_pair_in_block + shl_pair0];
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[64] = yjyi;
            rjri[128] = zjzi;
            rjri[192] = rr_ij;
        }
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double s0, s1, s2;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            int ijp = ijkp / kprim;
            int kp = ijkp % kprim;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[64] * aj_aij + ri[1];
            double zij = rjri[128] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[192];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            double theta_rr = theta * rr;
            if (omega == 0) {
                rys_roots(3, theta_rr, rw, 64, gout_id, 4);
            } else if (omega > 0) {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                __syncthreads();
                double sqrt_theta_fac = sqrt(theta_fac);
                for (int irys = gout_id; irys < 3; irys+=4) {
                    rw[ irys*2   *64] *= theta_fac;
                    rw[(irys*2+1)*64] *= sqrt_theta_fac;
                }
            } else {
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double *rw1 = rw + 384;
                rys_roots(3, theta_rr, rw1, 64, gout_id, 4);
                rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                __syncthreads();
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = gout_id; irys < 3; irys+=4) {
                    rw[ irys*2   *64] *= theta_fac;
                    rw[(irys*2+1)*64] *= sqrt_theta_fac;
                }
            }
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*128];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                double b01 = .5/ak * (1 - rt_ak);
                for (int n = gout_id; n < 3; n += 4) {
                    if (n == 2) {
                        gz[0] = rw[irys*128+64];
                    }
                    double *_gx = gx + n * 1152;
                    double xjxi = rjri[n * 64];
                    double Rpa = xjxi * aj_aij;
                    double c0x = Rpa - rt_aij * Rpq[n * 64];
                    s0 = _gx[0];
                    s1 = c0x * s0;
                    _gx[64] = s1;
                    s2 = c0x * s1 + 1 * b10 * s0;
                    _gx[128] = s2;
                    s0 = s1;
                    s1 = s2;
                    s2 = c0x * s1 + 2 * b10 * s0;
                    _gx[192] = s2;
                    double cpx = rt_ak * Rpq[n * 64];
                    s0 = _gx[0];
                    s1 = cpx * s0;
                    _gx[384] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    _gx[768] = s2;
                    s0 = _gx[64];
                    s1 = cpx * s0;
                    s1 += 1 * b00 * _gx[0];
                    _gx[448] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    s2 += 1 * b00 * _gx[384];
                    _gx[832] = s2;
                    s0 = _gx[128];
                    s1 = cpx * s0;
                    s1 += 2 * b00 * _gx[64];
                    _gx[512] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    s2 += 2 * b00 * _gx[448];
                    _gx[896] = s2;
                    s0 = _gx[192];
                    s1 = cpx * s0;
                    s1 += 3 * b00 * _gx[128];
                    _gx[576] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    s2 += 3 * b00 * _gx[512];
                    _gx[960] = s2;
                    s1 = _gx[192];
                    s0 = _gx[128];
                    _gx[320] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[64];
                    _gx[256] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[0];
                    _gx[192] = s1 - xjxi * s0;
                    s1 = _gx[576];
                    s0 = _gx[512];
                    _gx[704] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[448];
                    _gx[640] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[384];
                    _gx[576] = s1 - xjxi * s0;
                    s1 = _gx[960];
                    s0 = _gx[896];
                    _gx[1088] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[832];
                    _gx[1024] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[768];
                    _gx[960] = s1 - xjxi * s0;
                }
                __syncthreads();
                switch (gout_id) {
                case 0:
                gout0 += gx[1088] * gy[0] * gz[0];
                gout1 += gx[320] * gy[384] * gz[384];
                gout2 += gx[640] * gy[64] * gz[384];
                gout3 += gx[1024] * gy[0] * gz[64];
                gout4 += gx[256] * gy[384] * gz[448];
                gout5 += gx[576] * gy[128] * gz[384];
                gout6 += gx[960] * gy[64] * gz[64];
                gout7 += gx[192] * gy[448] * gz[448];
                gout8 += gx[576] * gy[0] * gz[512];
                gout9 += gx[896] * gy[192] * gz[0];
                gout10 += gx[128] * gy[576] * gz[384];
                gout11 += gx[448] * gy[256] * gz[384];
                gout12 += gx[832] * gy[192] * gz[64];
                gout13 += gx[64] * gy[576] * gz[448];
                gout14 += gx[384] * gy[320] * gz[384];
                gout15 += gx[768] * gy[256] * gz[64];
                gout16 += gx[0] * gy[640] * gz[448];
                gout17 += gx[384] * gy[192] * gz[512];
                gout18 += gx[896] * gy[0] * gz[192];
                gout19 += gx[128] * gy[384] * gz[576];
                gout20 += gx[448] * gy[64] * gz[576];
                gout21 += gx[832] * gy[0] * gz[256];
                gout22 += gx[64] * gy[384] * gz[640];
                gout23 += gx[384] * gy[128] * gz[576];
                gout24 += gx[768] * gy[64] * gz[256];
                gout25 += gx[0] * gy[448] * gz[640];
                gout26 += gx[384] * gy[0] * gz[704];
                break;
                case 1:
                gout0 += gx[704] * gy[384] * gz[0];
                gout1 += gx[320] * gy[0] * gz[768];
                gout2 += gx[256] * gy[832] * gz[0];
                gout3 += gx[640] * gy[384] * gz[64];
                gout4 += gx[256] * gy[0] * gz[832];
                gout5 += gx[192] * gy[896] * gz[0];
                gout6 += gx[576] * gy[448] * gz[64];
                gout7 += gx[192] * gy[64] * gz[832];
                gout8 += gx[192] * gy[768] * gz[128];
                gout9 += gx[512] * gy[576] * gz[0];
                gout10 += gx[128] * gy[192] * gz[768];
                gout11 += gx[64] * gy[1024] * gz[0];
                gout12 += gx[448] * gy[576] * gz[64];
                gout13 += gx[64] * gy[192] * gz[832];
                gout14 += gx[0] * gy[1088] * gz[0];
                gout15 += gx[384] * gy[640] * gz[64];
                gout16 += gx[0] * gy[256] * gz[832];
                gout17 += gx[0] * gy[960] * gz[128];
                gout18 += gx[512] * gy[384] * gz[192];
                gout19 += gx[128] * gy[0] * gz[960];
                gout20 += gx[64] * gy[832] * gz[192];
                gout21 += gx[448] * gy[384] * gz[256];
                gout22 += gx[64] * gy[0] * gz[1024];
                gout23 += gx[0] * gy[896] * gz[192];
                gout24 += gx[384] * gy[448] * gz[256];
                gout25 += gx[0] * gy[64] * gz[1024];
                gout26 += gx[0] * gy[768] * gz[320];
                break;
                case 2:
                gout0 += gx[704] * gy[0] * gz[384];
                gout1 += gx[1024] * gy[64] * gz[0];
                gout2 += gx[256] * gy[448] * gz[384];
                gout3 += gx[640] * gy[0] * gz[448];
                gout4 += gx[960] * gy[128] * gz[0];
                gout5 += gx[192] * gy[512] * gz[384];
                gout6 += gx[576] * gy[64] * gz[448];
                gout7 += gx[960] * gy[0] * gz[128];
                gout8 += gx[192] * gy[384] * gz[512];
                gout9 += gx[512] * gy[192] * gz[384];
                gout10 += gx[832] * gy[256] * gz[0];
                gout11 += gx[64] * gy[640] * gz[384];
                gout12 += gx[448] * gy[192] * gz[448];
                gout13 += gx[768] * gy[320] * gz[0];
                gout14 += gx[0] * gy[704] * gz[384];
                gout15 += gx[384] * gy[256] * gz[448];
                gout16 += gx[768] * gy[192] * gz[128];
                gout17 += gx[0] * gy[576] * gz[512];
                gout18 += gx[512] * gy[0] * gz[576];
                gout19 += gx[832] * gy[64] * gz[192];
                gout20 += gx[64] * gy[448] * gz[576];
                gout21 += gx[448] * gy[0] * gz[640];
                gout22 += gx[768] * gy[128] * gz[192];
                gout23 += gx[0] * gy[512] * gz[576];
                gout24 += gx[384] * gy[64] * gz[640];
                gout25 += gx[768] * gy[0] * gz[320];
                gout26 += gx[0] * gy[384] * gz[704];
                break;
                case 3:
                gout0 += gx[320] * gy[768] * gz[0];
                gout1 += gx[640] * gy[448] * gz[0];
                gout2 += gx[256] * gy[64] * gz[768];
                gout3 += gx[256] * gy[768] * gz[64];
                gout4 += gx[576] * gy[512] * gz[0];
                gout5 += gx[192] * gy[128] * gz[768];
                gout6 += gx[192] * gy[832] * gz[64];
                gout7 += gx[576] * gy[384] * gz[128];
                gout8 += gx[192] * gy[0] * gz[896];
                gout9 += gx[128] * gy[960] * gz[0];
                gout10 += gx[448] * gy[640] * gz[0];
                gout11 += gx[64] * gy[256] * gz[768];
                gout12 += gx[64] * gy[960] * gz[64];
                gout13 += gx[384] * gy[704] * gz[0];
                gout14 += gx[0] * gy[320] * gz[768];
                gout15 += gx[0] * gy[1024] * gz[64];
                gout16 += gx[384] * gy[576] * gz[128];
                gout17 += gx[0] * gy[192] * gz[896];
                gout18 += gx[128] * gy[768] * gz[192];
                gout19 += gx[448] * gy[448] * gz[192];
                gout20 += gx[64] * gy[64] * gz[960];
                gout21 += gx[64] * gy[768] * gz[256];
                gout22 += gx[384] * gy[512] * gz[192];
                gout23 += gx[0] * gy[128] * gz[960];
                gout24 += gx[0] * gy[832] * gz[256];
                gout25 += gx[384] * gy[384] * gz[320];
                gout26 += gx[0] * gy[0] * gz[1088];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            int *ao_loc = envs.ao_loc;
            int k0 = ao_loc[ksh0] - ao_loc[nbas];
            double *eri_tensor = out_local + shl_pair_in_block * 18 * naux + k0 + ksh_in_block * 6;
            switch (gout_id) {
            case 0:
            eri_tensor[0*naux + 0] = gout0;
            eri_tensor[0*naux + 4] = gout1;
            eri_tensor[1*naux + 2] = gout2;
            eri_tensor[2*naux + 0] = gout3;
            eri_tensor[2*naux + 4] = gout4;
            eri_tensor[3*naux + 2] = gout5;
            eri_tensor[4*naux + 0] = gout6;
            eri_tensor[4*naux + 4] = gout7;
            eri_tensor[5*naux + 2] = gout8;
            eri_tensor[6*naux + 0] = gout9;
            eri_tensor[6*naux + 4] = gout10;
            eri_tensor[7*naux + 2] = gout11;
            eri_tensor[8*naux + 0] = gout12;
            eri_tensor[8*naux + 4] = gout13;
            eri_tensor[9*naux + 2] = gout14;
            eri_tensor[10*naux + 0] = gout15;
            eri_tensor[10*naux + 4] = gout16;
            eri_tensor[11*naux + 2] = gout17;
            eri_tensor[12*naux + 0] = gout18;
            eri_tensor[12*naux + 4] = gout19;
            eri_tensor[13*naux + 2] = gout20;
            eri_tensor[14*naux + 0] = gout21;
            eri_tensor[14*naux + 4] = gout22;
            eri_tensor[15*naux + 2] = gout23;
            eri_tensor[16*naux + 0] = gout24;
            eri_tensor[16*naux + 4] = gout25;
            eri_tensor[17*naux + 2] = gout26;
            break;
            case 1:
            eri_tensor[0*naux + 1] = gout0;
            eri_tensor[0*naux + 5] = gout1;
            eri_tensor[1*naux + 3] = gout2;
            eri_tensor[2*naux + 1] = gout3;
            eri_tensor[2*naux + 5] = gout4;
            eri_tensor[3*naux + 3] = gout5;
            eri_tensor[4*naux + 1] = gout6;
            eri_tensor[4*naux + 5] = gout7;
            eri_tensor[5*naux + 3] = gout8;
            eri_tensor[6*naux + 1] = gout9;
            eri_tensor[6*naux + 5] = gout10;
            eri_tensor[7*naux + 3] = gout11;
            eri_tensor[8*naux + 1] = gout12;
            eri_tensor[8*naux + 5] = gout13;
            eri_tensor[9*naux + 3] = gout14;
            eri_tensor[10*naux + 1] = gout15;
            eri_tensor[10*naux + 5] = gout16;
            eri_tensor[11*naux + 3] = gout17;
            eri_tensor[12*naux + 1] = gout18;
            eri_tensor[12*naux + 5] = gout19;
            eri_tensor[13*naux + 3] = gout20;
            eri_tensor[14*naux + 1] = gout21;
            eri_tensor[14*naux + 5] = gout22;
            eri_tensor[15*naux + 3] = gout23;
            eri_tensor[16*naux + 1] = gout24;
            eri_tensor[16*naux + 5] = gout25;
            eri_tensor[17*naux + 3] = gout26;
            break;
            case 2:
            eri_tensor[0*naux + 2] = gout0;
            eri_tensor[1*naux + 0] = gout1;
            eri_tensor[1*naux + 4] = gout2;
            eri_tensor[2*naux + 2] = gout3;
            eri_tensor[3*naux + 0] = gout4;
            eri_tensor[3*naux + 4] = gout5;
            eri_tensor[4*naux + 2] = gout6;
            eri_tensor[5*naux + 0] = gout7;
            eri_tensor[5*naux + 4] = gout8;
            eri_tensor[6*naux + 2] = gout9;
            eri_tensor[7*naux + 0] = gout10;
            eri_tensor[7*naux + 4] = gout11;
            eri_tensor[8*naux + 2] = gout12;
            eri_tensor[9*naux + 0] = gout13;
            eri_tensor[9*naux + 4] = gout14;
            eri_tensor[10*naux + 2] = gout15;
            eri_tensor[11*naux + 0] = gout16;
            eri_tensor[11*naux + 4] = gout17;
            eri_tensor[12*naux + 2] = gout18;
            eri_tensor[13*naux + 0] = gout19;
            eri_tensor[13*naux + 4] = gout20;
            eri_tensor[14*naux + 2] = gout21;
            eri_tensor[15*naux + 0] = gout22;
            eri_tensor[15*naux + 4] = gout23;
            eri_tensor[16*naux + 2] = gout24;
            eri_tensor[17*naux + 0] = gout25;
            eri_tensor[17*naux + 4] = gout26;
            break;
            case 3:
            eri_tensor[0*naux + 3] = gout0;
            eri_tensor[1*naux + 1] = gout1;
            eri_tensor[1*naux + 5] = gout2;
            eri_tensor[2*naux + 3] = gout3;
            eri_tensor[3*naux + 1] = gout4;
            eri_tensor[3*naux + 5] = gout5;
            eri_tensor[4*naux + 3] = gout6;
            eri_tensor[5*naux + 1] = gout7;
            eri_tensor[5*naux + 5] = gout8;
            eri_tensor[6*naux + 3] = gout9;
            eri_tensor[7*naux + 1] = gout10;
            eri_tensor[7*naux + 5] = gout11;
            eri_tensor[8*naux + 3] = gout12;
            eri_tensor[9*naux + 1] = gout13;
            eri_tensor[9*naux + 5] = gout14;
            eri_tensor[10*naux + 3] = gout15;
            eri_tensor[11*naux + 1] = gout16;
            eri_tensor[11*naux + 5] = gout17;
            eri_tensor[12*naux + 3] = gout18;
            eri_tensor[13*naux + 1] = gout19;
            eri_tensor[13*naux + 5] = gout20;
            eri_tensor[14*naux + 3] = gout21;
            eri_tensor[15*naux + 1] = gout22;
            eri_tensor[15*naux + 5] = gout23;
            eri_tensor[16*naux + 3] = gout24;
            eri_tensor[17*naux + 1] = gout25;
            eri_tensor[17*naux + 5] = gout26;
            break;
            }
        }
    }
}

__device__
int int3c2e_bdiv_unrolled(double *out, Int3c2eEnvVars envs, BDiv3c2eBounds bounds)
{
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int shl_pair0 = bounds.shl_pair_offsets[sp_block_id];
    int ksh0 = bounds.ksh_offsets[ksh_block_id];
    int bas_ij0 = bounds.bas_ij_idx[shl_pair0];
    int nbas = envs.nbas;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;
    int *bas = envs.bas;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    int lk = bas[ksh0*BAS_SLOTS+ANG_OF];
    int kij_type = lk*25 + li*5 + lj;
    switch (kij_type) {
    case 0: int3c2e_bdiv_000(out, envs, bounds); break;
    case 5: int3c2e_bdiv_100(out, envs, bounds); break;
    case 6: int3c2e_bdiv_110(out, envs, bounds); break;
    case 10: int3c2e_bdiv_200(out, envs, bounds); break;
    case 11: int3c2e_bdiv_210(out, envs, bounds); break;
    case 12: int3c2e_bdiv_220(out, envs, bounds); break;
    case 25: int3c2e_bdiv_001(out, envs, bounds); break;
    case 30: int3c2e_bdiv_101(out, envs, bounds); break;
    case 31: int3c2e_bdiv_111(out, envs, bounds); break;
    case 35: int3c2e_bdiv_201(out, envs, bounds); break;
    case 36: int3c2e_bdiv_211(out, envs, bounds); break;
    case 37: int3c2e_bdiv_221(out, envs, bounds); break;
    case 50: int3c2e_bdiv_002(out, envs, bounds); break;
    case 55: int3c2e_bdiv_102(out, envs, bounds); break;
    case 56: int3c2e_bdiv_112(out, envs, bounds); break;
    case 60: int3c2e_bdiv_202(out, envs, bounds); break;
    case 61: int3c2e_bdiv_212(out, envs, bounds); break;
    default: return 0;
    }
    return 1;
}
