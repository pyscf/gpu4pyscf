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
#include "pbc.cuh"
#include "ft_ao.cuh"
#include "gvhf-rys/rys_contract_k.cuh"

#define NGV_PER_BLOCK   32
#define NSP_PER_BLOCK   8
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787
#define OF_COMPLEX      2

__device__ __forceinline__
void multiply(double aR, double aI, double bR, double bI, double &cR, double &cI)
{
    double outR = aR * bR - aI * bI;
    double outI = aR * bI + aI * bR;
    cR = outR;
    cI = outI;
}

__global__
void ft_aopair_ejk_ip1_kernel(double *out, double *dm, double *vG, double *Gv,
                              PBCIntEnvVars envs, int nGv, int shm_size,
                              int *bas_ij_idx, int *bas_ij_img_idx,
                              int *shl_pair_offsets)
{
    constexpr int nGv_per_block = NGV_PER_BLOCK;
    constexpr int threads = NGV_PER_BLOCK * NSP_PER_BLOCK;
    int sp_block_id = blockIdx.x;
    int Gv_block_id = blockIdx.y;
    int Gv_id_in_block = threadIdx.x;

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int shl_pair0 = shl_pair_offsets[sp_block_id];
    int shl_pair1 = shl_pair_offsets[sp_block_id+1];
    int bas_ij0 = bas_ij_idx[shl_pair0];
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    int stride_j = li + 2;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nGv_per_block * NSP_PER_BLOCK;
    int gout_stride = 1;
    while (8*6*gx_len > shm_size) {
        gx_len /= 2;
        gout_stride *= 2;
    }
    int nsp_per_block = NSP_PER_BLOCK / gout_stride;
    int gout_id = threadIdx.y % gout_stride;
    int sp_id = threadIdx.y / gout_stride;
    int Gv_gout_id = Gv_id_in_block + nGv_per_block * gout_id;
    int nGv_gout = nGv_per_block * gout_stride;
    int lij = li + lj + 1;
    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int i_1 =          nGv_per_block;
    int j_1 = stride_j*nGv_per_block;
    int *ao_loc = envs.ao_loc;
    int nao = ao_loc[envs.cell0_nbas];

    int Gv_id = Gv_block_id * nGv_per_block + Gv_id_in_block;
    Gv += Gv_id;
    double kx = Gv[0];
    double ky = Gv[nGv];
    double kz = Gv[nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;

    extern __shared__ double shared_memory[];
    double *gxR = shared_memory + g_size * nGv_per_block * sp_id + Gv_id_in_block;
    double *gxI = gxR + gx_len*1;
    double *gyR = gxR + gx_len*2;
    double *gyI = gxR + gx_len*3;
    double *gzR = gxR + gx_len*4;
    double *gzI = gxR + gx_len*5;
    double *rjri = shared_memory + gx_len * 6 + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        __syncthreads();
        int bas_ij, jL;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
            jL = bas_ij_img_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
            jL = bas_ij_img_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.cell0_nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (Gv_gout_id == 0) {
            double xjxi = rj[0] + img_coords[jL*3+0] - ri[0];
            double yjyi = rj[1] + img_coords[jL*3+1] - ri[1];
            double zjzi = rj[2] + img_coords[jL*3+2] - ri[2];
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        // Note the density matrix is assumed to be real in get_ej_ip1 function
        double *dm_ij;
        if (vG == NULL) {
            dm_ij = dm + (Gv_id + (j0*nao+i0) * nGv) * OF_COMPLEX;
        } else {
            dm_ij = dm + (j0*nao+i0);
        }

        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double s0xR, s1xR, s2xR;
        double s0xI, s1xI, s2xI;

        for (int ijp = 0; ijp < ijprim; ++ijp) {
            __syncthreads();
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ai2 = ai * 2;
            double aj2 = aj * 2;
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double a2 = .5 / aij;
            if (gout_id == 0) {
                double theta_ij = ai * aj_aij;
                double fac = OVERLAP_FAC * ci[ip] * cj[jp] / (aij * sqrt(aij));
                if (ish_cell0 == jsh_cell0) {
                    fac *= .5;
                }
                if (Gv_id >= nGv) {
                    fac = 0;
                }
                double xjxi = rjri[0*nsp_per_block];
                double yjyi = rjri[1*nsp_per_block];
                double zjzi = rjri[2*nsp_per_block];
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double kR = kx * xij + ky * yij + kz * zij;
                sincos(-kR, gzI, gzR);
                double rr = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_rr = theta_ij*rr + .5*a2*kk;
                double Kab = exp(-theta_rr);
                gxR[0] = fac;
                gxI[0] = 0.;
                gyR[0] = 1.;
                gyI[0] = 0.;
                // exp(-theta_rr-kR*1j)
                gzR[0] *= Kab;
                gzI[0] *= Kab;
            }

            // gx[i+1] = ia2 * gx[i-1] + (rijrx[0] - kx[n]*a2*_Complex_I) * gx[i];
            __syncthreads();
            for (int n = gout_id; n < 3; n += gout_stride) {
                double *_gxR = gxR + n * gx_len * OF_COMPLEX;
                double *_gxI = _gxR + gx_len;
                double RpaR = rjri[n*nsp_per_block] * aj_aij; // Rp - Ra
                double RpaI = -a2 * Gv[nGv*n];
                s0xR = _gxR[0];
                s0xI = _gxI[0];
                multiply(RpaR, RpaI, s0xR, s0xI, s1xR, s1xI);
                _gxR[nGv_per_block] = s1xR;
                _gxI[nGv_per_block] = s1xI;
                for (int i = 1; i < lij; i++) {
                    double ia2 = i * a2;
                    multiply(RpaR, RpaI, s1xR, s1xI, s2xR, s2xI);
                    s2xR += ia2 * s0xR;
                    s2xI += ia2 * s0xI;
                    _gxR[(i+1)*nGv_per_block] = s2xR;
                    _gxI[(i+1)*nGv_per_block] = s2xI;
                    s0xR = s1xR;
                    s0xI = s1xI;
                    s1xR = s2xR;
                    s1xI = s2xI;
                }
            }

            // hrr
            if (lj > 0) {
                __syncthreads();
                for (int n = gout_id; n < 3*OF_COMPLEX; n += gout_stride) {
                    double *_gx = gxR + n * gx_len;
                    // The real and imaginary parts call the same expression
                    int _ix = n / 2;
                    double xjxi = rjri[_ix*nsp_per_block];
                    for (int j = 0; j < lj; ++j) {
                        int ij = (lij-j) + j*stride_j;
                        s1xR = _gx[ij*nGv_per_block];
                        for (--ij; ij >= j*stride_j; --ij) {
                            s0xR = _gx[ij*nGv_per_block];
                            _gx[(ij+stride_j)*nGv_per_block] = s1xR - xjxi * s0xR;
                            s1xR = s0xR;
                        }
                    }
                }
            }
            __syncthreads();
            if (pair_ij >= shl_pair1 || Gv_id >= nGv) {
                continue;
            }
            for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                int i = ij % nfi;
                int j = ij / nfi;
                double dm_vR, dm_vI;
                if (vG == NULL) {
                    int addr = (j*nao+i)*nGv * OF_COMPLEX;
                    dm_vR = dm_ij[addr];
                    dm_vI = dm_ij[addr+1];
                } else {
                    double tmp = dm_ij[j*nao+i];
                    dm_vR = tmp * vG[Gv_id*OF_COMPLEX  ];
                    dm_vI = tmp * vG[Gv_id*OF_COMPLEX+1];
                }
                int ix = idx_i[i*3+0];
                int iy = idx_i[i*3+1];
                int iz = idx_i[i*3+2];
                int jx = idx_j[j*3+0];
                int jy = idx_j[j*3+1];
                int jz = idx_j[j*3+2];
                int addrx = (ix + jx*stride_j) * nGv_per_block;
                int addry = (iy + jy*stride_j) * nGv_per_block;
                int addrz = (iz + jz*stride_j) * nGv_per_block;
                double IxR = gxR[addrx];
                double IxI = gxI[addrx];
                double IyR = gyR[addry];
                double IyI = gyI[addry];
                double IzR = gzR[addrz];
                double IzI = gzI[addrz];
                double prod_xyR, prod_xyI;
                double prod_xzR, prod_xzI;
                double prod_yzR, prod_yzI;
                multiply(IxR, IxI, IyR, IyI, prod_xyR, prod_xyI);
                multiply(IxR, IxI, IzR, IzI, prod_xzR, prod_xzI);
                multiply(IyR, IyI, IzR, IzI, prod_yzR, prod_yzI);
                multiply(prod_xyR, prod_xyI, dm_vR, dm_vI, prod_xyR, prod_xyI);
                multiply(prod_xzR, prod_xzI, dm_vR, dm_vI, prod_xzR, prod_xzI);
                multiply(prod_yzR, prod_yzI, dm_vR, dm_vI, prod_yzR, prod_yzI);
                double gixR = gxR[addrx+i_1];
                double gixI = gxI[addrx+i_1];
                double giyR = gyR[addry+i_1];
                double giyI = gyI[addry+i_1];
                double gizR = gzR[addrz+i_1];
                double gizI = gzI[addrz+i_1];
                double fjxR = aj2 * (gixR - rjri[0*nsp_per_block] * IxR);
                double fjxI = aj2 * (gixI - rjri[0*nsp_per_block] * IxI);
                double fjyR = aj2 * (giyR - rjri[1*nsp_per_block] * IyR);
                double fjyI = aj2 * (giyI - rjri[1*nsp_per_block] * IyI);
                double fjzR = aj2 * (gizR - rjri[2*nsp_per_block] * IzR);
                double fjzI = aj2 * (gizI - rjri[2*nsp_per_block] * IzI);
                if (jx > 0) { fjxR -= jx * gxR[addrx-j_1]; fjxI -= jx * gxI[addrx-j_1]; }
                if (jy > 0) { fjyR -= jy * gyR[addry-j_1]; fjyI -= jy * gyI[addry-j_1]; }
                if (jz > 0) { fjzR -= jz * gzR[addrz-j_1]; fjzI -= jz * gzI[addrz-j_1]; }
                v_jx += fjxR * prod_yzR - fjxI * prod_yzI;
                v_jy += fjyR * prod_xzR - fjyI * prod_xzI;
                v_jz += fjzR * prod_xyR - fjzI * prod_xyI;
                double fixR = ai2 * gixR;
                double fiyR = ai2 * giyR;
                double fizR = ai2 * gizR;
                double fixI = ai2 * gixI;
                double fiyI = ai2 * giyI;
                double fizI = ai2 * gizI;
                if (ix > 0) { fixR -= ix * gxR[addrx-i_1]; fixI -= ix * gxI[addrx-i_1]; }
                if (iy > 0) { fiyR -= iy * gyR[addry-i_1]; fiyI -= iy * gyI[addry-i_1]; }
                if (iz > 0) { fizR -= iz * gzR[addrz-i_1]; fizI -= iz * gzI[addrz-i_1]; }
                v_ix += fixR * prod_yzR - fixI * prod_yzI;
                v_iy += fiyR * prod_xzR - fiyI * prod_xzI;
                v_iz += fizR * prod_xyR - fizI * prod_xyI;
            }
        }

        double *reduce = shared_memory + thread_id;
        __syncthreads();
        reduce[0*threads] = v_ix;
        reduce[1*threads] = v_iy;
        reduce[2*threads] = v_iz;
        reduce[3*threads] = v_jx;
        reduce[4*threads] = v_jy;
        reduce[5*threads] = v_jz;
        for (int i = nGv_gout/2; i > 0; i >>= 1) {
            __syncthreads();
            if (Gv_gout_id < i) {
#pragma unroll
                for (int n = 0; n < 6; ++n) {
                    reduce[n*threads] += reduce[n*threads+i];
                }
            }
        }
        if (Gv_gout_id == 0 && pair_ij < shl_pair1) {
            int ia = bas[ish_cell0*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh_cell0*BAS_SLOTS+ATOM_OF];
            atomicAdd(out+ia*3+0, reduce[0*threads]);
            atomicAdd(out+ia*3+1, reduce[1*threads]);
            atomicAdd(out+ia*3+2, reduce[2*threads]);
            atomicAdd(out+ja*3+0, reduce[3*threads]);
            atomicAdd(out+ja*3+1, reduce[4*threads]);
            atomicAdd(out+ja*3+2, reduce[5*threads]);
        }
    }
}

__global__
void ft_aopair_strain_deriv_kernel(double *out, double *sigma,
                              double *dm, double *vG, double *Gv,
                              PBCIntEnvVars envs, int nGv, int shm_size,
                              int *bas_ij_idx, int *bas_ij_img_idx,
                              int *shl_pair_offsets)
{
    constexpr int nGv_per_block = NGV_PER_BLOCK;
    constexpr int threads = NGV_PER_BLOCK * NSP_PER_BLOCK;
    int sp_block_id = blockIdx.x;
    int Gv_block_id = blockIdx.y;
    int Gv_id_in_block = threadIdx.x;

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int shl_pair0 = shl_pair_offsets[sp_block_id];
    int shl_pair1 = shl_pair_offsets[sp_block_id+1];
    int bas_ij0 = bas_ij_idx[shl_pair0];
    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int ish0 = bas_ij0 / nbas;
    int jsh0 = bas_ij0 % nbas;

    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    int stride_j = li + 2;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nGv_per_block * NSP_PER_BLOCK;
    int gout_stride = 1;
    while (8*6*gx_len > shm_size) {
        gx_len /= 2;
        gout_stride *= 2;
    }
    int nsp_per_block = NSP_PER_BLOCK / gout_stride;
    int gout_id = threadIdx.y % gout_stride;
    int sp_id = threadIdx.y / gout_stride;
    int Gv_gout_id = Gv_id_in_block + nGv_per_block * gout_id;
    int nGv_gout = nGv_per_block * gout_stride;
    int lij = li + lj + 1;
    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int nfij = nfi * nfj;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int i_1 =          nGv_per_block;
    int j_1 = stride_j*nGv_per_block;
    int *ao_loc = envs.ao_loc;
    int nao = ao_loc[envs.cell0_nbas];

    int Gv_id = Gv_block_id * nGv_per_block + Gv_id_in_block;
    Gv += Gv_id;
    double kx = Gv[0];
    double ky = Gv[nGv];
    double kz = Gv[nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;

    extern __shared__ double shared_memory[];
    double *gxR = shared_memory + g_size * nGv_per_block * sp_id + Gv_id_in_block;
    double *gxI = gxR + gx_len*1;
    double *gyR = gxR + gx_len*2;
    double *gyI = gxR + gx_len*3;
    double *gzR = gxR + gx_len*4;
    double *gzI = gxR + gx_len*5;
    double *rjri = shared_memory + gx_len * 6 + sp_id;
    int *idx_i = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    int *idx_j = _c_cartesian_lexical_xyz + lex_xyz_offset(lj);

    double sigma_xx = 0;
    double sigma_xy = 0;
    double sigma_xz = 0;
    double sigma_yx = 0;
    double sigma_yy = 0;
    double sigma_yz = 0;
    double sigma_zx = 0;
    double sigma_zy = 0;
    double sigma_zz = 0;

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        __syncthreads();
        int bas_ij, jL;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
            jL = bas_ij_img_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
            jL = bas_ij_img_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.cell0_nbas;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double xj = rj[0] + img_coords[jL*3+0];
        double yj = rj[1] + img_coords[jL*3+1];
        double zj = rj[2] + img_coords[jL*3+2];
        if (Gv_gout_id == 0) {
            double xjxi = xj - xi;
            double yjyi = yj - yi;
            double zjzi = zj - zi;
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        // Note the density matrix is assumed to be real in get_ej_ip1 function
        double *dm_ij;
        if (vG == NULL) {
            dm_ij = dm + (Gv_id + (j0*nao+i0) * nGv) * OF_COMPLEX;
        } else {
            dm_ij = dm + (j0*nao+i0);
        }

        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double s0xR, s1xR, s2xR;
        double s0xI, s1xI, s2xI;
        double goutx, gouty, goutz;

        for (int ijp = 0; ijp < ijprim; ++ijp) {
            __syncthreads();
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double ai2 = ai * 2;
            double aj2 = aj * 2;
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double a2 = .5 / aij;
            if (gout_id == 0) {
                double theta_ij = ai * aj_aij;
                double fac = OVERLAP_FAC * ci[ip] * cj[jp] / (aij * sqrt(aij));
                if (ish_cell0 == jsh_cell0) {
                    fac *= .5;
                }
                if (Gv_id >= nGv) {
                    fac = 0;
                }
                double xjxi = rjri[0*nsp_per_block];
                double yjyi = rjri[1*nsp_per_block];
                double zjzi = rjri[2*nsp_per_block];
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double kR = kx * xij + ky * yij + kz * zij;
                sincos(-kR, gzI, gzR);
                double rr = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_rr = theta_ij*rr + .5*a2*kk;
                double Kab = exp(-theta_rr);
                gxR[0] = fac;
                gxI[0] = 0.;
                gyR[0] = 1.;
                gyI[0] = 0.;
                // exp(-theta_rr-kR*1j)
                gzR[0] *= Kab;
                gzI[0] *= Kab;
            }

            // gx[i+1] = ia2 * gx[i-1] + (rijrx[0] - kx*a2*_Complex_I) * gx[i];
            __syncthreads();
            for (int n = gout_id; n < 3; n += gout_stride) {
                double *_gxR = gxR + n * gx_len * OF_COMPLEX;
                double *_gxI = _gxR + gx_len;
                double RpaR = rjri[n*nsp_per_block] * aj_aij; // Rp - Ra
                double RpaI = -a2 * Gv[nGv*n];
                s0xR = _gxR[0];
                s0xI = _gxI[0];
                multiply(RpaR, RpaI, s0xR, s0xI, s1xR, s1xI);
                _gxR[nGv_per_block] = s1xR;
                _gxI[nGv_per_block] = s1xI;
                for (int i = 1; i < lij; i++) {
                    double ia2 = i * a2;
                    multiply(RpaR, RpaI, s1xR, s1xI, s2xR, s2xI);
                    s2xR += ia2 * s0xR;
                    s2xI += ia2 * s0xI;
                    _gxR[(i+1)*nGv_per_block] = s2xR;
                    _gxI[(i+1)*nGv_per_block] = s2xI;
                    s0xR = s1xR;
                    s0xI = s1xI;
                    s1xR = s2xR;
                    s1xI = s2xI;
                }
            }

            // hrr
            if (lj > 0) {
                __syncthreads();
                for (int n = gout_id; n < 3*OF_COMPLEX; n += gout_stride) {
                    double *_gx = gxR + n * gx_len;
                    // The real and imaginary parts call the same expression
                    int _ix = n / 2;
                    double xjxi = rjri[_ix*nsp_per_block];
                    for (int j = 0; j < lj; ++j) {
                        int ij = (lij-j) + j*stride_j;
                        s1xR = _gx[ij*nGv_per_block];
                        for (--ij; ij >= j*stride_j; --ij) {
                            s0xR = _gx[ij*nGv_per_block];
                            _gx[(ij+stride_j)*nGv_per_block] = s1xR - xjxi * s0xR;
                            s1xR = s0xR;
                        }
                    }
                }
            }
            __syncthreads();
            if (pair_ij >= shl_pair1 || Gv_id >= nGv) {
                continue;
            }
            for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                int i = ij % nfi;
                int j = ij / nfi;
                double dm_vR, dm_vI;
                if (vG == NULL) {
                    int addr = (j*nao+i)*nGv * OF_COMPLEX;
                    dm_vR = dm_ij[addr];
                    dm_vI = dm_ij[addr+1];
                } else {
                    double tmp = dm_ij[j*nao+i];
                    dm_vR = tmp * vG[Gv_id*OF_COMPLEX  ];
                    dm_vI = tmp * vG[Gv_id*OF_COMPLEX+1];
                }
                int ix = idx_i[i*3+0];
                int iy = idx_i[i*3+1];
                int iz = idx_i[i*3+2];
                int jx = idx_j[j*3+0];
                int jy = idx_j[j*3+1];
                int jz = idx_j[j*3+2];
                int addrx = (ix + jx*stride_j) * nGv_per_block;
                int addry = (iy + jy*stride_j) * nGv_per_block;
                int addrz = (iz + jz*stride_j) * nGv_per_block;
                double IxR = gxR[addrx];
                double IxI = gxI[addrx];
                double IyR = gyR[addry];
                double IyI = gyI[addry];
                double IzR = gzR[addrz];
                double IzI = gzI[addrz];
                double prod_xyR, prod_xyI;
                double prod_xzR, prod_xzI;
                double prod_yzR, prod_yzI;
                multiply(IxR, IxI, IyR, IyI, prod_xyR, prod_xyI);
                multiply(IxR, IxI, IzR, IzI, prod_xzR, prod_xzI);
                multiply(IyR, IyI, IzR, IzI, prod_yzR, prod_yzI);
                multiply(prod_xyR, prod_xyI, dm_vR, dm_vI, prod_xyR, prod_xyI);
                multiply(prod_xzR, prod_xzI, dm_vR, dm_vI, prod_xzR, prod_xzI);
                multiply(prod_yzR, prod_yzI, dm_vR, dm_vI, prod_yzR, prod_yzI);
                double gixR = gxR[addrx+i_1];
                double gixI = gxI[addrx+i_1];
                double giyR = gyR[addry+i_1];
                double giyI = gyI[addry+i_1];
                double gizR = gzR[addrz+i_1];
                double gizI = gzI[addrz+i_1];
                // <i|exp(-iGr)|\nabla j>
                double fjxR = aj2 * (gixR - rjri[0*nsp_per_block] * IxR);
                double fjxI = aj2 * (gixI - rjri[0*nsp_per_block] * IxI);
                double fjyR = aj2 * (giyR - rjri[1*nsp_per_block] * IyR);
                double fjyI = aj2 * (giyI - rjri[1*nsp_per_block] * IyI);
                double fjzR = aj2 * (gizR - rjri[2*nsp_per_block] * IzR);
                double fjzI = aj2 * (gizI - rjri[2*nsp_per_block] * IzI);
                if (jx > 0) { fjxR -= jx * gxR[addrx-j_1]; fjxI -= jx * gxI[addrx-j_1]; }
                if (jy > 0) { fjyR -= jy * gyR[addry-j_1]; fjyI -= jy * gyI[addry-j_1]; }
                if (jz > 0) { fjzR -= jz * gzR[addrz-j_1]; fjzI -= jz * gzI[addrz-j_1]; }
                goutx = fjxR * prod_yzR - fjxI * prod_yzI;
                gouty = fjyR * prod_xzR - fjyI * prod_xzI;
                goutz = fjzR * prod_xyR - fjzI * prod_xyI;
                v_jx += goutx;
                v_jy += gouty;
                v_jz += goutz;
                sigma_xx += goutx * xj;
                sigma_xy += goutx * yj;
                sigma_xz += goutx * zj;
                sigma_yx += gouty * xj;
                sigma_yy += gouty * yj;
                sigma_yz += gouty * zj;
                sigma_zx += goutz * xj;
                sigma_zy += goutz * yj;
                sigma_zz += goutz * zj;
                // <\nabla i|exp(-iGr)|j>
                double fixR = ai2 * gixR;
                double fiyR = ai2 * giyR;
                double fizR = ai2 * gizR;
                double fixI = ai2 * gixI;
                double fiyI = ai2 * giyI;
                double fizI = ai2 * gizI;
                if (ix > 0) { fixR -= ix * gxR[addrx-i_1]; fixI -= ix * gxI[addrx-i_1]; }
                if (iy > 0) { fiyR -= iy * gyR[addry-i_1]; fiyI -= iy * gyI[addry-i_1]; }
                if (iz > 0) { fizR -= iz * gzR[addrz-i_1]; fizI -= iz * gzI[addrz-i_1]; }
                goutx = fixR * prod_yzR - fixI * prod_yzI;
                gouty = fiyR * prod_xzR - fiyI * prod_xzI;
                goutz = fizR * prod_xyR - fizI * prod_xyI;
                v_ix += goutx;
                v_iy += gouty;
                v_iz += goutz;
                sigma_xx += goutx * xi;
                sigma_xy += goutx * yi;
                sigma_xz += goutx * zi;
                sigma_yx += gouty * xi;
                sigma_yy += gouty * yi;
                sigma_yz += gouty * zi;
                sigma_zx += goutz * xi;
                sigma_zy += goutz * yi;
                sigma_zz += goutz * zi;
                // <i|\nabla_(e_ij) exp(-iGr)|j> = <i|-iy exp(-iGr)|j> Gx
                //   = -i <(y-Yi + Yi)i|exp(-iGr)|j> Gx
                goutx = (gixR + xi * IxR) * prod_yzI + (gixI + xi * IxI) * prod_yzR;
                gouty = (giyR + yi * IyR) * prod_xzI + (giyI + yi * IyI) * prod_xzR;
                goutz = (gizR + zi * IzR) * prod_xyI + (gizI + zi * IzI) * prod_xyR;
                sigma_xx -= kx * goutx;
                sigma_xy -= kx * gouty;
                sigma_xz -= kx * goutz;
                sigma_yx -= ky * goutx;
                sigma_yy -= ky * gouty;
                sigma_yz -= ky * goutz;
                sigma_zx -= kz * goutx;
                sigma_zy -= kz * gouty;
                sigma_zz -= kz * goutz;
            }
        }

        double *reduce = shared_memory + thread_id;
        __syncthreads();
        reduce[0*threads] = v_ix;
        reduce[1*threads] = v_iy;
        reduce[2*threads] = v_iz;
        reduce[3*threads] = v_jx;
        reduce[4*threads] = v_jy;
        reduce[5*threads] = v_jz;
        for (int i = nGv_gout/2; i > 0; i >>= 1) {
            __syncthreads();
            if (Gv_gout_id < i) {
#pragma unroll
                for (int n = 0; n < 6; ++n) {
                    reduce[n*threads] += reduce[n*threads+i];
                }
            }
        }
        if (Gv_gout_id == 0 && pair_ij < shl_pair1) {
            int ia = bas[ish_cell0*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh_cell0*BAS_SLOTS+ATOM_OF];
            atomicAdd(out+ia*3+0, reduce[0*threads]);
            atomicAdd(out+ia*3+1, reduce[1*threads]);
            atomicAdd(out+ia*3+2, reduce[2*threads]);
            atomicAdd(out+ja*3+0, reduce[3*threads]);
            atomicAdd(out+ja*3+1, reduce[4*threads]);
            atomicAdd(out+ja*3+2, reduce[5*threads]);
        }
    }
    atomicAdd(sigma+0, sigma_xx);
    atomicAdd(sigma+1, sigma_xy);
    atomicAdd(sigma+2, sigma_xz);
    atomicAdd(sigma+3, sigma_yx);
    atomicAdd(sigma+4, sigma_yy);
    atomicAdd(sigma+5, sigma_yz);
    atomicAdd(sigma+6, sigma_zx);
    atomicAdd(sigma+7, sigma_zy);
    atomicAdd(sigma+8, sigma_zz);
}

extern "C" {
int PBC_ft_aopair_ej_ip1(double *out, double *dm, double *vG, double *GvT,
                         PBCIntEnvVars *envs,
                         int nbatches_shl_pair, int ngrids, int shm_size,
                         int *bas_ij_idx, int *bas_ij_img_idx, int *shl_pair_offsets,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    cudaFuncSetAttribute(ft_aopair_ejk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 threads(NGV_PER_BLOCK, NSP_PER_BLOCK);
    int Gv_batches = (ngrids + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    dim3 blocks(nbatches_shl_pair, Gv_batches);
    ft_aopair_ejk_ip1_kernel<<<blocks, threads, shm_size>>>(
            out, dm, vG, GvT, *envs, ngrids, shm_size,
            bas_ij_idx, bas_ij_img_idx, shl_pair_offsets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_ej_ip1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBC_ft_aopair_ek_ip1(double *out, double *dm_vG, double *GvT, PBCIntEnvVars *envs,
                         int nbatches_shl_pair, int ngrids, int shm_size,
                         int *bas_ij_idx, int *bas_ij_img_idx, int *shl_pair_offsets,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    cudaFuncSetAttribute(ft_aopair_ejk_ip1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 threads(NGV_PER_BLOCK, NSP_PER_BLOCK);
    int Gv_batches = (ngrids + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    dim3 blocks(nbatches_shl_pair, Gv_batches);
    ft_aopair_ejk_ip1_kernel<<<blocks, threads, shm_size>>>(
            out, dm_vG, NULL, GvT, *envs, ngrids, shm_size,
            bas_ij_idx, bas_ij_img_idx, shl_pair_offsets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_ek_ip1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBC_ft_aopair_ej_strain_deriv(double *out, double *sigma, double *dm,
                         double *vG, double *GvT, PBCIntEnvVars *envs,
                         int nbatches_shl_pair, int ngrids, int shm_size,
                         int *bas_ij_idx, int *bas_ij_img_idx, int *shl_pair_offsets,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    cudaFuncSetAttribute(ft_aopair_strain_deriv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 threads(NGV_PER_BLOCK, NSP_PER_BLOCK);
    int Gv_batches = (ngrids + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    dim3 blocks(nbatches_shl_pair, Gv_batches);
    ft_aopair_strain_deriv_kernel<<<blocks, threads, shm_size>>>(
            out, sigma, dm, vG, GvT, *envs, ngrids, shm_size,
            bas_ij_idx, bas_ij_img_idx, shl_pair_offsets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_ej_strain_deriv: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBC_ft_aopair_ek_strain_deriv(double *out, double *sigma,
                         double *dm_vG, double *GvT, PBCIntEnvVars *envs,
                         int nbatches_shl_pair, int ngrids, int shm_size,
                         int *bas_ij_idx, int *bas_ij_img_idx, int *shl_pair_offsets,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    cudaFuncSetAttribute(ft_aopair_strain_deriv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 threads(NGV_PER_BLOCK, NSP_PER_BLOCK);
    int Gv_batches = (ngrids + NGV_PER_BLOCK - 1) / NGV_PER_BLOCK;
    dim3 blocks(nbatches_shl_pair, Gv_batches);
    ft_aopair_strain_deriv_kernel<<<blocks, threads, shm_size>>>(
            out, sigma, dm_vG, NULL, GvT, *envs, ngrids, shm_size,
            bas_ij_idx, bas_ij_img_idx, shl_pair_offsets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_ek_strain_deriv: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
