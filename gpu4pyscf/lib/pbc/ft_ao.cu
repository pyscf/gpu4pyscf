/*
 * Copyright 2024 The PySCF Developers. All Rights Reserved.
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
#include "ft_ao.cuh"

#define GOUT_WIDTH      19
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787
#define OF_COMPLEX      2

__global__
void ft_aopair_kernel(double *out, AFTIntEnvVars envs, AFTBoundsInfo bounds)
{
    // sp is short for shl_pair
    int sp_block_id = blockIdx.x;
    int Gv_block_id = blockIdx.y;
    int nGv_per_block = blockDim.x;
    int gout_stride = blockDim.y;
    int nsp_per_block = blockDim.z;
    int Gv_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_id = threadIdx.z;
    int npairs_ij = bounds.npairs_ij;
    int pair_ij_idx = sp_block_id * nsp_per_block + sp_id;

    if (pair_ij_idx >= npairs_ij) {
        return;
    }

    int nbas = envs.nbas;
    int ish = bounds.ish_in_pair[pair_ij_idx];
    int jsh = bounds.jsh_in_pair[pair_ij_idx];
    int *sp_img_offsets = envs.img_offsets;
    int bas_ij = ish * nbas + jsh;
    int img0 = sp_img_offsets[bas_ij];
    int img1 = sp_img_offsets[bas_ij+1];
    if (img0 >= img1) {
        return;
    }

    int li = bounds.li;
    int lj = bounds.lj;
    int nfij = bounds.nfij;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int ijprim = iprim * jprim;
    int lij = li + lj;
    int stride_j = bounds.stride_j;
    int g_size = bounds.g_size;
    int gx_len = g_size * nGv_per_block * nsp_per_block;
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    int ia = bas[ish*BAS_SLOTS+ATOM_OF];
    int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
    double *ri = env + atm[ia*ATM_SLOTS+PTR_COORD];
    double *rj = env + atm[ja*ATM_SLOTS+PTR_COORD];
    double *img_coords = envs.img_coords;
    int *img_idx = envs.img_idx;

    int nGv = bounds.ngrids;
    double *Gv = bounds.grids + Gv_block_id * nGv_per_block;
    double kx = Gv[Gv_id];
    double ky = Gv[Gv_id + nGv];
    double kz = Gv[Gv_id + nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;
    double rjri[3];

    extern __shared__ double g[];
    double *gxR = g + g_size * nGv_per_block * sp_id + Gv_id;
    double *gxI = gxR + gx_len;
    double *gyR = gxI + gx_len;
    double *gyI = gyR + gx_len;
    double *gzR = gyI + gx_len;
    double *gzI = gzR + gx_len;

    double goutR[GOUT_WIDTH];
    double goutI[GOUT_WIDTH];
#pragma unroll
    for (int n = 0; n < GOUT_WIDTH; ++n) {
        goutR[n] = 0.;
        goutI[n] = 0.;
    }
    double s0xR, s1xR, s2xR;
    double s0xI, s1xI, s2xI;

    for (int ijp = 0; ijp < ijprim; ++ijp) {
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double theta_ij = ai * aj_aij;
        double a2 = .5 / aij;
        double fac = OVERLAP_FAC * ci[ip] * cj[jp] / (aij * sqrt(aij));

        for (int img = img0; img < img1; img++) {
            int img_id = img_idx[img];
            double Lx = img_coords[img_id*3+0];
            double Ly = img_coords[img_id*3+1];
            double Lz = img_coords[img_id*3+2];
            double xjxi = rj[0] + Lx - ri[0];
            double yjyi = rj[1] + Ly - ri[1];
            double zjzi = rj[2] + Lz - ri[2];
            rjri[0] = xjxi;
            rjri[1] = yjyi;
            rjri[2] = zjzi;

            __syncthreads();
            if (gout_id == 0) {
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

            if (lij > 0) {
                // gx[i+1] = ia2 * gx[i-1] + (rijrx[0] - kx[n]*a2*_Complex_I) * gx[i];
                __syncthreads();
                for (int n = gout_id; n < 3; n += gout_stride) {
                    double *_gxR = gxR + n * gx_len * OF_COMPLEX;
                    double *_gxI = _gxR + gx_len;
                    double RpaR = rjri[n] * aj_aij; // Rp - Ra
                    double RpaI = -a2 * Gv[Gv_id+nGv*n];
                    s0xR = _gxR[0];
                    s0xI = _gxI[0];
                    s1xR = RpaR * s0xR - RpaI * s0xI;
                    s1xI = RpaR * s0xI + RpaI * s0xR;
                    _gxR[nGv_per_block] = s1xR;
                    _gxI[nGv_per_block] = s1xI;
                    for (int i = 1; i < lij; i++) {
                        double ia2 = i * a2;
                        s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
                        s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
                        _gxR[(i+1)*nGv_per_block] = s2xR;
                        _gxI[(i+1)*nGv_per_block] = s2xI;
                        s0xR = s1xR;
                        s0xI = s1xI;
                        s1xR = s2xR;
                        s1xI = s2xI;
                    }
                }
            }

            // hrr
            if (lj > 0) {
                __syncthreads();
                for (int n = gout_id; n < 3*OF_COMPLEX; n += gout_stride) {
                    double *_gx = gxR + n * gx_len;
                    // The real and imaginary parts call the same expression
                    int _ix = n / 2;
                    double xjxi = rjri[_ix];
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
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride + gout_id;
                if (ij >= nfij) break;
                int addrx = idx_ij[ij] * nGv_per_block;
                int addry = idy_ij[ij] * nGv_per_block;
                int addrz = idz_ij[ij] * nGv_per_block;
                double xR = gxR[addrx];
                double xI = gxI[addrx];
                double yR = gyR[addry];
                double yI = gyI[addry];
                double zR = gzR[addrz];
                double zI = gzI[addrz];
                double xyR = xR * yR - xI * yI;
                double xyI = xR * yI + xI * yR;
                goutR[n] += xyR * zR - xyI * zI;
                goutI[n] += xyR * zI + xyI * zR;
            }
        }
    }

    if (Gv_block_id * nGv_per_block + Gv_id < nGv) {
        int nfi = (li + 1) * (li + 2) / 2;
        int *ao_loc = envs.ao_loc;
        int ncells = envs.bvk_ncells;
        int nbasp = nbas / ncells;
        size_t nao = ao_loc[nbasp];
        size_t cell_id = jsh / nbasp;
        int cell0_jsh = jsh % nbasp;
        size_t i0 = ao_loc[ish];
        size_t j0 = ao_loc[cell0_jsh];
        double *aft_tensor = out + 
                (cell_id * nao*nao*nGv + (i0*nao+j0) * nGv
                 + Gv_block_id*nGv_per_block + Gv_id) * OF_COMPLEX;
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            int ij = n*gout_stride + gout_id;
            if (ij >= nfij) break;
            size_t i = ij % nfi;
            size_t j = ij / nfi;
            size_t addr = (i*nao+j)*nGv;
            aft_tensor[addr*2  ] = goutR[n];
            aft_tensor[addr*2+1] = goutI[n];
        }
    }
}

__global__
void ft_aopair_fill_triu(double *out, int *conj_mapping, int bvk_ncells, int nGv)
{
    int j = blockIdx.x;
    int i = blockIdx.y;
    if (i <= j) {
        return;
    }
    size_t nao = gridDim.x;
    size_t nao2_nGv = nao * nao * nGv;
    size_t ij = (i * nao + j) * nGv;
    size_t ji = (j * nao + i) * nGv;
    for (int n = threadIdx.x; n < bvk_ncells*nGv; n += blockDim.x) {
        int Gv_id = n % nGv;
        int k = n / nGv;
        int ck = conj_mapping[k];
        out[ji + ck*nao2_nGv+Gv_id] = out[ij + k*nao2_nGv+Gv_id];
    }
}
