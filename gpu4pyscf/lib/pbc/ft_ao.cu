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
void ft_aopair_kernel(double *out, AFTIntEnvVars envs, AFTBoundsInfo bounds,
                      int compressing)
{
    // sp is short for shl_pair
    int sp_block_id = blockIdx.x;
    int Gv_block_id = blockIdx.y;
    int nGv_per_block = blockDim.x;
    int gout_stride = blockDim.y;
    int nsp_per_block = blockDim.z;
    int Gv_id_in_block = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_id = threadIdx.z;
    int npairs_ij = bounds.npairs_ij;
    int pair_ij = sp_block_id * nsp_per_block + sp_id;
    if (pair_ij >= npairs_ij) {
        return;
    }

    int nbas = envs.cell0_nbas * envs.bvk_ncells;
    int bas_ij = bounds.bas_ij_idx[pair_ij];
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int *sp_img_offsets = bounds.img_offsets;
    int img0 = sp_img_offsets[pair_ij];
    int img1 = sp_img_offsets[pair_ij+1];

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
    int *img_idx = bounds.img_idx;

    int nGv = bounds.ngrids;
    int Gv_id = Gv_block_id * nGv_per_block + Gv_id_in_block;
    double *Gv = bounds.grids + Gv_id;
    double kx = Gv[0];
    double ky = Gv[nGv];
    double kz = Gv[nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;
    double rjri[3];

    extern __shared__ double g[];
    double *gxR = g + g_size * nGv_per_block * sp_id + Gv_id_in_block;
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
                    double RpaI = -a2 * Gv[nGv*n];
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

    if (Gv_id < nGv) {
        if (compressing) {
            int nfi = (li + 1) * (li + 2) / 2;
            int nfj = (lj + 1) * (lj + 2) / 2;
            int nfij = nfi * nfj;
            int stride = npairs_ij * nGv * OF_COMPLEX;
            double *aft_tensor = out + (pair_ij * nGv + Gv_id) * OF_COMPLEX;
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ij = n*gout_stride + gout_id;
                if (ij >= nfij) break;
                aft_tensor[ij*stride  ] = goutR[n];
                aft_tensor[ij*stride+1] = goutI[n];
            }
        } else {
            int nfi = (li + 1) * (li + 2) / 2;
            int *ao_loc = envs.ao_loc;
            int nbasp = envs.cell0_nbas;
            size_t nao = ao_loc[nbasp];
            size_t cell_id = jsh / nbasp;
            int cell0_jsh = jsh % nbasp;
            size_t i0 = ao_loc[ish];
            size_t j0 = ao_loc[cell0_jsh];
            double *aft_tensor = out +
                    (cell_id * nao*nao*nGv + (i0*nao+j0) * nGv + Gv_id) * OF_COMPLEX;
#pragma unroll
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

//__global__
//void ft_aopair_unpack(double *out, double *dat, int *addresses,
//                       int *conj_mapping, int bvk_ncells, int nGv)
//{
//}

#define REMOTE_THRESHOLD 50

// count images for the overlap between cell and bvkcell
__global__ static
void overlap_img_counts_kernel(int *img_counts, int ish0, int jsh0, int nish, int njsh,
                               AFTIntEnvVars envs, float *exps, float *log_coeff,
                               float log_cutoff, int permutation_symmetry)
{
    int bas_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int s_njsh = envs.bvk_ncells * njsh;
    if (bas_ij >= nish*s_njsh) {
        return;
    }
    int nimgs = envs.nimgs;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int ish = bas_ij / s_njsh;
    int jsh = bas_ij % s_njsh;
    int cell0_ish = ish        + ish0;;
    int cell0_jsh = jsh % njsh + jsh0;;
    if (permutation_symmetry && cell0_ish < cell0_jsh) {
        return;
    }
    ish =                                cell0_ish;
    jsh = jsh / njsh * envs.cell0_nbas + cell0_jsh;
    int li = bas[ANG_OF + cell0_ish*BAS_SLOTS];
    int lj = bas[ANG_OF + cell0_jsh*BAS_SLOTS];
    float ai = exps[cell0_ish];
    float aj = exps[cell0_jsh];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta_ij = ai * aj / aij;
    float log_ci = log_coeff[cell0_ish];
    float log_cj = log_coeff[cell0_jsh];
    float log_cicj = log_ci + log_cj;
    double *ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    double *rj = env + atm[bas[jsh*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    // log(ci*cj * (pi/aij)**1.5)
    float log_fac = log_cicj + 1.717f - 1.5f*logf(aij);
    log_cutoff = log_cutoff - log_fac;

    int counts = 0;
    for (int img = 0; img < nimgs; ++img) {
        float xjL = xj + img_coords[img*3+0];
        float yjL = yj + img_coords[img*3+1];
        float zjL = zj + img_coords[img*3+2];
        float xjxi = xjL - xi;
        float yjyi = yjL - yi;
        float zjzi = zjL - zi;
        float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
        float theta_ij_rr = theta_ij * rr_ij;
        if (theta_ij_rr > REMOTE_THRESHOLD) {
            continue;
        }

        float dr = sqrtf(rr_ij);
        float dri = fj * dr;
        float drj = fi * dr;
        float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
        float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
        float estimator = dri_fac + drj_fac + theta_ij_rr;
        if (estimator > log_cutoff) {
            counts++;
        }
    }
    img_counts[bas_ij] = counts;
}

__global__ static
void overlap_img_idx_kernel(int *img_idx, int *img_offsets, int *bas_ij_mapping,
                            int npairs, int ish0, int jsh0, int nish, int njsh,
                            AFTIntEnvVars envs, float *exps, float *log_coeff,
                            float log_cutoff)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= npairs) {
        return;
    }
    int bas_ij = bas_ij_mapping[pair_id];
    int s_njsh = envs.bvk_ncells * njsh;
    int ish = bas_ij / s_njsh;
    int jsh = bas_ij % s_njsh;
    int cell0_ish = ish        + ish0;;
    int cell0_jsh = jsh % njsh + jsh0;;
    ish =                                cell0_ish;
    jsh = jsh / njsh * envs.cell0_nbas + cell0_jsh;

    int nimgs = envs.nimgs;
    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int li = bas[ANG_OF + cell0_ish*BAS_SLOTS];
    int lj = bas[ANG_OF + cell0_jsh*BAS_SLOTS];
    float ai = exps[cell0_ish];
    float aj = exps[cell0_jsh];
    float aij = ai + aj;
    float fi = ai / aij;
    float fj = aj / aij;
    float theta_ij = ai * aj / aij;
    float log_ci = log_coeff[cell0_ish];
    float log_cj = log_coeff[cell0_jsh];
    float log_cicj = log_ci + log_cj;
    double *ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    double *rj = env + atm[bas[jsh*BAS_SLOTS+ATOM_OF] * ATM_SLOTS + PTR_COORD];
    float xi = ri[0];
    float yi = ri[1];
    float zi = ri[2];
    float xj = rj[0];
    float yj = rj[1];
    float zj = rj[2];
    // log(ci*cj * (pi/aij)**1.5)
    float log_fac = log_cicj + 1.717f - 1.5f*logf(aij);
    log_cutoff = log_cutoff - log_fac;

    int counts = 0;
    img_idx += img_offsets[pair_id];
    for (int img = 0; img < nimgs; ++img) {
        float xjL = xj + img_coords[img*3+0];
        float yjL = yj + img_coords[img*3+1];
        float zjL = zj + img_coords[img*3+2];
        float xjxi = xjL - xi;
        float yjyi = yjL - yi;
        float zjzi = zjL - zi;
        float rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
        float theta_ij_rr = theta_ij * rr_ij;
        if (theta_ij_rr > REMOTE_THRESHOLD) {
            continue;
        }

        float dr = sqrtf(rr_ij);
        float dri = fj * dr;
        float drj = fi * dr;
        float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
        float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
        float estimator = dri_fac + drj_fac + theta_ij_rr;
        if (estimator > log_cutoff) {
            img_idx[counts] = img;
            counts++;
        }
    }
}

extern "C" {
int overlap_img_counts(int *img_counts, int *shls_slice, AFTIntEnvVars *envs,
                       float *exps, float *log_coeff, float log_cutoff,
                       int permutation_symmetry)
{
    int ish0 = shls_slice[0];
    int ish1 = shls_slice[1];
    int jsh0 = shls_slice[2];
    int jsh1 = shls_slice[3];
    int nish = ish1 - ish0;
    int njsh = jsh1 - jsh0;
    constexpr int threads = 512;
    int ncells = envs->bvk_ncells;
    int blocks = (nish*ncells*njsh + threads-1)/threads;
    overlap_img_counts_kernel<<<blocks, threads>>>(
        img_counts, ish0, jsh0, nish, njsh, *envs, exps, log_coeff, log_cutoff,
        permutation_symmetry);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in overlap_img_counts: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int overlap_img_idx(int *img_idx, int *img_offsets, int *bas_ij_mapping,
                    int npairs, int *shls_slice, AFTIntEnvVars *envs,
                    float *exps, float *log_coeff, float log_cutoff)
{
    int ish0 = shls_slice[0];
    int ish1 = shls_slice[1];
    int jsh0 = shls_slice[2];
    int jsh1 = shls_slice[3];
    int nish = ish1 - ish0;
    int njsh = jsh1 - jsh0;
    constexpr int threads = 512;
    int blocks = (npairs + threads-1)/threads;
    overlap_img_idx_kernel<<<blocks, threads>>>(
        img_idx, img_offsets, bas_ij_mapping, npairs, ish0, jsh0, nish, njsh,
        *envs, exps, log_coeff, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in overlap_img_counts: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
