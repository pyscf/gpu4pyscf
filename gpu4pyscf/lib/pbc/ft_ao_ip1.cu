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
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_contract_k.cuh"
#include "pbc.cuh"
#include "ft_ao.cuh"

#define NSP_PER_BLOCK   8

__device__ __forceinline__
void multiply(double aR, double aI, double bR, double bI, double &cR, double &cI)
{
    double outR = aR * bR - aI * bI;
    double outI = aR * bI + aI * bR;
    cR = outR;
    cI = outI;
}

// inlcuding both nuclear gradients and strain derivatives
__global__
void ft_aopair_deriv_kernel(double *grad, double *sigma,
                            double *dm, double2 *vG, double *Gv,
                            PBCIntEnvVars envs, int nGv, int shm_size,
                            int *bas_ij_idx, int *bas_ij_img_idx,
                            int *shl_pair_offsets, int permutation_symmetry)
{
    constexpr int nGv_per_block = NG_PER_BLOCK;
    constexpr int threads = NG_PER_BLOCK * NSP_PER_BLOCK;
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
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int ijprim = iprim * jprim;
    int i_1 =          nGv_per_block;
    int j_1 = stride_j*nGv_per_block;
    int *ao_loc = envs.ao_loc;
    int nao = ao_loc[envs.cell0_nbas];

    int Gv_id = Gv_block_id * nGv_per_block + Gv_id_in_block;
    double kx = 0;
    double ky = 0;
    double kz = 0;
    if (Gv_id < nGv) {
        kx = Gv[Gv_id];
        ky = Gv[Gv_id + nGv];
        kz = Gv[Gv_id + nGv * 2];
    }
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
            dm_ij = dm + (Gv_id + (size_t)(j0*nao+i0) * nGv) * OF_COMPLEX;
        } else {
            dm_ij = dm + (j0*nao+i0);
        }

        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double goutx = 0;
        double gouty = 0;
        double goutz = 0;
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
                if (permutation_symmetry && ish_cell0 == jsh_cell0) {
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
                double RpaI = -a2;
                if (Gv_id < nGv) {
                    RpaI *= Gv[Gv_id+nGv*n];
                }
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
            float div_nfi = c_div_nf[li];
            for (int ij = gout_id; ij < nfij; ij += gout_stride) {
                uint32_t j = ij * div_nfi;
                uint32_t i = ij - nfi * j;
                double dm_vR, dm_vI;
                if (vG == NULL) {
                    size_t addr = (size_t)(j*nao+i)*nGv * OF_COMPLEX;
                    dm_vR = dm_ij[addr];
                    dm_vI = dm_ij[addr+1];
                } else {
                    double tmp = dm_ij[j*nao+i];
                    double2 tmp_vG = vG[Gv_id];
                    dm_vR = tmp * tmp_vG.x;
                    dm_vI = tmp * tmp_vG.y;
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
                v_jx += fjxR * prod_yzR - fjxI * prod_yzI;
                v_jy += fjyR * prod_xzR - fjyI * prod_xzI;
                v_jz += fjzR * prod_xyR - fjzI * prod_xyI;
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
                v_ix += fixR * prod_yzR - fixI * prod_yzI;
                v_iy += fiyR * prod_xzR - fiyI * prod_xzI;
                v_iz += fizR * prod_xyR - fizI * prod_xyI;
                // <i|\nabla_(e_xy) exp(-iGr)|j> = <i|-iy exp(-iGr)|j> Gx
                //   = -i <(y-Yi + Yi)i|exp(-iGr)|j> Gx
                goutx += (gixR + xi * IxR) * prod_yzI + (gixI + xi * IxI) * prod_yzR;
                gouty += (giyR + yi * IyR) * prod_xzI + (giyI + yi * IyI) * prod_xzR;
                goutz += (gizR + zi * IzR) * prod_xyI + (gizI + zi * IzI) * prod_xyR;
            }
        }
        sigma_xx += v_ix * xi + v_jx * xj - kx * goutx;
        sigma_xy += v_ix * yi + v_jx * yj - kx * gouty;
        sigma_xz += v_ix * zi + v_jx * zj - kx * goutz;
        sigma_yx += v_iy * xi + v_jy * xj - ky * goutx;
        sigma_yy += v_iy * yi + v_jy * yj - ky * gouty;
        sigma_yz += v_iy * zi + v_jy * zj - ky * goutz;
        sigma_zx += v_iz * xi + v_jz * xj - kz * goutx;
        sigma_zy += v_iz * yi + v_jz * yj - kz * gouty;
        sigma_zz += v_iz * zi + v_jz * zj - kz * goutz;

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
            atomicAdd(grad+ia*3+0, reduce[0*threads]);
            atomicAdd(grad+ia*3+1, reduce[1*threads]);
            atomicAdd(grad+ia*3+2, reduce[2*threads]);
            atomicAdd(grad+ja*3+0, reduce[3*threads]);
            atomicAdd(grad+ja*3+1, reduce[4*threads]);
            atomicAdd(grad+ja*3+2, reduce[5*threads]);
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

// inlcuding both nuclear gradients and strain derivatives
__global__ static
void ft_ao_deriv_kernel(double *grad, double *sigma, double2 *dm_auxG, double *Gv,
                        RysIntEnvVars envs, int nGv)
{
    int sh_block_id = gridDim.y - blockIdx.y - 1;
    int Gv_block_id = blockIdx.x;
    int nsh_per_block = FT_AO_THREADS / NG_PER_BLOCK;
    int sh_id_in_block = threadIdx.y;
    int Gv_id_in_block = threadIdx.x;
    int sh_id = sh_block_id * nsh_per_block + sh_id_in_block;
    if (sh_id >= envs.nbas) {
        return;
    }

    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    int li = bas[sh_id*BAS_SLOTS+ANG_OF];
    int Gv_id = Gv_block_id * NG_PER_BLOCK + Gv_id_in_block;
    double kx = 0;
    double ky = 0;
    double kz = 0;
    if (Gv_id < nGv) {
        kx = Gv[Gv_id];
        ky = Gv[Gv_id + nGv];
        kz = Gv[Gv_id + nGv * 2];
    }
    double kk = kx * kx + ky * ky + kz * kz;

    constexpr int aux_l = AUXL + 1;
    int gx_len = (aux_l+1) * FT_AO_THREADS;
    int i_1 =  NG_PER_BLOCK;
    __shared__ double g[(aux_l+1)*FT_AO_THREADS * 6];
    double *gxR = g + (aux_l+1) * NG_PER_BLOCK * sh_id_in_block + Gv_id_in_block;
    double *gxI = gxR + gx_len;
    double *gyR = gxR + gx_len*2;
    double *gyI = gxR + gx_len*3;
    double *gzR = gxR + gx_len*4;
    double *gzI = gxR + gx_len*5;
    int *idx = _c_cartesian_lexical_xyz + lex_xyz_offset(li);
    double s0xR, s1xR, s2xR;
    double s0xI, s1xI, s2xI;
    double s0yR, s1yR, s2yR;
    double s0yI, s1yI, s2yI;
    double s0zR, s1zR, s2zR;
    double s0zI, s1zI, s2zI;
    double v_Gx = 0;
    double v_Gy = 0;
    double v_Gz = 0;
    double prod = 0;

    int ia = bas[sh_id*BAS_SLOTS+ATOM_OF];
    int expi = bas[sh_id*BAS_SLOTS+PTR_EXP];
    int ci = bas[sh_id*BAS_SLOTS+PTR_COEFF];
    int ri = atm[ia*ATM_SLOTS+PTR_COORD];
    double xi = env[ri+0];
    double yi = env[ri+1];
    double zi = env[ri+2];
    int i0 = envs.ao_loc[sh_id];
    int iprim = bas[sh_id*BAS_SLOTS+NPRIM_OF];
    for (int ip = 0; ip < iprim; ++ip) {
        __syncthreads();
        double ai = env[expi+ip];
        double kR = kx * xi + ky * yi + kz * zi;
        sincos(-kR, &s0zI, &s0zR);
        double Kab = exp(-.25/ai*kk);
        s0xR = OVERLAP_FAC * env[ci+ip] / (ai * sqrt(ai));
        s0xI = 0.;
        s0yR = 1.;
        s0yI = 0.;
        s0zR *= Kab;
        s0zI *= Kab;
        gxR[0] = s0xR;
        gxI[0] = s0xI;
        gyR[0] = s0yR;
        gyI[0] = s0yI;
        gzR[0] = s0zR;
        gzI[0] = s0zI;

        double a2 = .5 / ai;
        double xpaI = -a2 * kx;
        double ypaI = -a2 * ky;
        double zpaI = -a2 * kz;
        s1xR = -xpaI * s0xI;
        s1xI =  xpaI * s0xR;
        s1yR = -ypaI * s0yI;
        s1yI =  ypaI * s0yR;
        s1zR = -zpaI * s0zI;
        s1zI =  zpaI * s0zR;
        gxR[NG_PER_BLOCK] = s1xR;
        gxI[NG_PER_BLOCK] = s1xI;
        gyR[NG_PER_BLOCK] = s1yR;
        gyI[NG_PER_BLOCK] = s1yI;
        gzR[NG_PER_BLOCK] = s1zR;
        gzI[NG_PER_BLOCK] = s1zI;
        for (int i = 2; i <= aux_l; i++) {
            if (i > li+1) break;
            double ia2 = (i-1) * a2;
            s2xR = ia2 * s0xR - xpaI * s1xI;
            s2xI = ia2 * s0xI + xpaI * s1xR;
            s2yR = ia2 * s0yR - ypaI * s1yI;
            s2yI = ia2 * s0yI + ypaI * s1yR;
            s2zR = ia2 * s0zR - zpaI * s1zI;
            s2zI = ia2 * s0zI + zpaI * s1zR;
            gxR[i*NG_PER_BLOCK] = s2xR;
            gxI[i*NG_PER_BLOCK] = s2xI;
            gyR[i*NG_PER_BLOCK] = s2yR;
            gyI[i*NG_PER_BLOCK] = s2yI;
            gzR[i*NG_PER_BLOCK] = s2zR;
            gzI[i*NG_PER_BLOCK] = s2zI;
            s0xR = s1xR;
            s0xI = s1xI;
            s0yR = s1yR;
            s0yI = s1yI;
            s0zR = s1zR;
            s0zI = s1zI;
            s1xR = s2xR;
            s1xI = s2xI;
            s1yR = s2yR;
            s1yI = s2yI;
            s1zR = s2zR;
            s1zI = s2zI;
        }
        __syncthreads();
        if (Gv_id < nGv) {
            int nfi = c_nf[li];
#pragma unroll
            for (int n = 0; n < AUXNF; ++n) {
                if (n >= nfi) break;
                size_t addr = (i0+n) * (size_t)nGv + Gv_id;
                double2 dm_v = dm_auxG[addr];
                double dm_vR = dm_v.x;
                double dm_vI = -dm_v.y;
                int addrx = idx[n*3+0] * NG_PER_BLOCK;
                int addry = idx[n*3+1] * NG_PER_BLOCK;
                int addrz = idx[n*3+2] * NG_PER_BLOCK;
                double xR = gxR[addrx];
                double xI = gxI[addrx];
                double yR = gyR[addry];
                double yI = gyI[addry];
                double zR = gzR[addrz];
                double zI = gzI[addrz];
                double prod_xyR, prod_xyI;
                double prod_xzR, prod_xzI;
                double prod_yzR, prod_yzI;
                multiply(xR, xI, yR, yI, prod_xyR, prod_xyI);
                multiply(xR, xI, zR, zI, prod_xzR, prod_xzI);
                multiply(yR, yI, zR, zI, prod_yzR, prod_yzI);
                multiply(prod_xyR, prod_xyI, dm_vR, dm_vI, prod_xyR, prod_xyI);
                multiply(prod_xzR, prod_xzI, dm_vR, dm_vI, prod_xzR, prod_xzI);
                multiply(prod_yzR, prod_yzI, dm_vR, dm_vI, prod_yzR, prod_yzI);
                // Gradients ~ auxG*-1j*Gv * dm_auxG.conj()
                // prod = Re(auxG * dm_axuG.conj() * -1j)
                prod += prod_xyR * zI + prod_xyI * zR;
                // \nabla_(e_xy) exp(-iGr) = -iy exp(-iGr) Gx
                //   = -i (y-Yi + Yi) * exp(-iGr) Gx
                double gixR = gxR[addrx+i_1];
                double gixI = gxI[addrx+i_1];
                double giyR = gyR[addry+i_1];
                double giyI = gyI[addry+i_1];
                double gizR = gzR[addrz+i_1];
                double gizI = gzI[addrz+i_1];
                // naively, sigma_xx can be computed as
                // v_Gx += (gixR + xi * xR) * prod_yzI + (gixI + xi * xI) * prod_yzR;
                // v_Gy += (giyR + yi * yR) * prod_xzI + (giyI + yi * yI) * prod_xzR;
                // v_Gz += (gizR + zi * zR) * prod_xyI + (gizI + zi * zI) * prod_xyR;
                // sigma_xx = prod * kx * xi - kx * v_Gx;
                // sigma_xy = prod * kx * yi - kx * v_Gy;
                // sigma_xz = prod * kx * zi - kx * v_Gz;
                // sigma_yx = prod * ky * xi - ky * v_Gx;
                // sigma_yy = prod * ky * yi - ky * v_Gy;
                // sigma_yz = prod * ky * zi - ky * v_Gz;
                // sigma_zx = prod * kz * xi - kz * v_Gx;
                // sigma_zy = prod * kz * yi - kz * v_Gy;
                // sigma_zz = prod * kz * zi - kz * v_Gz;
                // However, prod * kx * xi can be cancelled, leaving the effective terms
                v_Gx += gixR * prod_yzI + gixI * prod_yzR;
                v_Gy += giyR * prod_xzI + giyI * prod_xzR;
                v_Gz += gizR * prod_xyI + gizI * prod_xyR;
            }
        }
    }
    double grad_x = prod * kx;
    double grad_y = prod * ky;
    double grad_z = prod * kz;
    //double sigma_xx = grad_x * xi - kx * v_Gx;
    //double sigma_xy = grad_x * yi - kx * v_Gy;
    //double sigma_xz = grad_x * zi - kx * v_Gz;
    //double sigma_yx = grad_y * xi - ky * v_Gx;
    //double sigma_yy = grad_y * yi - ky * v_Gy;
    //double sigma_yz = grad_y * zi - ky * v_Gz;
    //double sigma_zx = grad_z * xi - kz * v_Gx;
    //double sigma_zy = grad_z * yi - kz * v_Gy;
    //double sigma_zz = grad_z * zi - kz * v_Gz;
    double sigma_xx = - kx * v_Gx;
    double sigma_xy = - kx * v_Gy;
    double sigma_xz = - kx * v_Gz;
    double sigma_yx = - ky * v_Gx;
    double sigma_yy = - ky * v_Gy;
    double sigma_yz = - ky * v_Gz;
    double sigma_zx = - kz * v_Gx;
    double sigma_zy = - kz * v_Gy;
    double sigma_zz = - kz * v_Gz;
    atomicAdd(grad+ia*3+0, grad_x);
    atomicAdd(grad+ia*3+1, grad_y);
    atomicAdd(grad+ia*3+2, grad_z);
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
int PBC_ft_ao_deriv(double *grad, double *sigma, double2 *dm_auxG,
                    double *GvT, RysIntEnvVars *envs, int ngrids)
{
    int nsh_per_block = FT_AO_THREADS/NG_PER_BLOCK;
    dim3 threads(NG_PER_BLOCK, nsh_per_block);
    int nbatches_grids = (ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK;
    int nbatches_shls = (envs->nbas + nsh_per_block - 1) / nsh_per_block;
    dim3 blocks(nbatches_grids, nbatches_shls);
    ft_ao_deriv_kernel<<<blocks, threads>>>(
            grad, sigma, dm_auxG, GvT, *envs, ngrids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_ao_deriv: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBC_ft_aopair_ej_deriv(double *out, double *sigma, double *dm,
                         double2 *vG, double *GvT, PBCIntEnvVars *envs,
                         int nbatches_shl_pair, int ngrids, int shm_size,
                         int *bas_ij_idx, int *bas_ij_img_idx, int *shl_pair_offsets,
                         int permutation_symmetry)
{
    cudaFuncSetAttribute(ft_aopair_deriv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 threads(NG_PER_BLOCK, NSP_PER_BLOCK);
    int Gv_batches = (ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK;
    dim3 blocks(nbatches_shl_pair, Gv_batches);
    ft_aopair_deriv_kernel<<<blocks, threads, shm_size>>>(
            out, sigma, dm, vG, GvT, *envs, ngrids, shm_size,
            bas_ij_idx, bas_ij_img_idx, shl_pair_offsets, permutation_symmetry);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_ej_deriv: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int PBC_ft_aopair_ek_deriv(double *out, double *sigma,
                         double *dm_vG, double *GvT, PBCIntEnvVars *envs,
                         int nbatches_shl_pair, int ngrids, int shm_size,
                         int *bas_ij_idx, int *bas_ij_img_idx, int *shl_pair_offsets,
                         int permutation_symmetry)
{
    cudaFuncSetAttribute(ft_aopair_deriv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 threads(NG_PER_BLOCK, NSP_PER_BLOCK);
    int Gv_batches = (ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK;
    dim3 blocks(nbatches_shl_pair, Gv_batches);
    ft_aopair_deriv_kernel<<<blocks, threads, shm_size>>>(
            out, sigma, dm_vG, NULL, GvT, *envs, ngrids, shm_size,
            bas_ij_idx, bas_ij_img_idx, shl_pair_offsets, permutation_symmetry);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_ek_deriv: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
