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
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_contract_k.cuh"

#define WARP_SIZE       32
#define WARPS           8
#define NG_PER_BLOCK    WARP_SIZE
#define FT_AO_THREADS   (WARP_SIZE*4)
#define GOUT_WIDTH      29
// pi^1.5
#define OVERLAP_FAC     5.56832799683170787
#define OF_COMPLEX      2
#define POOL_SIZE       65536
#define AUXL            6
#define AUXNF           ((AUXL+1)*(AUXL+2)/2)

__global__ static
void ft_ao_bdiv_kernel(double *out, RysIntEnvVars envs, int nGv, double *grids)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();

    int sh_block_id = item.get_group_range(1) - item.get_group(1) - 1;
    int Gv_block_id = item.get_group(0);
    int sh_id_in_block = item.get_local_id(0);
    int Gv_id_in_block = item.get_local_id(1);

    double (&g)[(AUXL+1)*FT_AO_THREADS * 6] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[(AUXL+1)*FT_AO_THREADS * 6]>(item.get_group());

    #else
    int sh_block_id = gridDim.x - blockIdx.x - 1;
    int Gv_block_id = blockIdx.y;
    int sh_id_in_block = threadIdx.y;
    int Gv_id_in_block = threadIdx.x;

    __shared__ double g[(AUXL+1)*FT_AO_THREADS * 6];
    #endif

    int nsh_per_block = FT_AO_THREADS / NG_PER_BLOCK;
    int sh_id = sh_block_id * nsh_per_block + sh_id_in_block;
    if (sh_id >= envs.nbas) {
        return;
    }

    int *atm = envs.atm;
    int *bas = envs.bas;
    double *env = envs.env;
    int li = bas[sh_id*BAS_SLOTS+ANG_OF];
    int nfi = c_nf[li];
    int iprim = bas[sh_id*BAS_SLOTS+NPRIM_OF];
    int Gv_id = Gv_block_id * NG_PER_BLOCK + Gv_id_in_block;
    double *Gv = grids + Gv_id;
    double kx = Gv[0];
    double ky = Gv[nGv];
    double kz = Gv[nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;

    int gx_len = (AUXL+1) * FT_AO_THREADS;
    double *gxR = g + (AUXL+1) * NG_PER_BLOCK * sh_id_in_block + Gv_id_in_block;
    double *gxI = gxR + gx_len;
    double *gyR = gxR + gx_len*2;
    double *gyI = gxR + gx_len*3;
    double *gzR = gxR + gx_len*4;
    double *gzI = gxR + gx_len*5;
    const int *idx = _c_cartesian_lexical_xyz + lex_xyz_offset(li);

    constexpr int aux_nf = (AUXL+1)*(AUXL+2)/2;
    double goutR[aux_nf];
    double goutI[aux_nf];
#pragma unroll
    for (int n = 0; n < aux_nf; ++n) {
        goutR[n] = 0.;
        goutI[n] = 0.;
    }
    double s0xR, s1xR, s2xR;
    double s0xI, s1xI, s2xI;
    double s0yR, s1yR, s2yR;
    double s0yI, s1yI, s2yI;
    double s0zR, s1zR, s2zR;
    double s0zI, s1zI, s2zI;

    int ia = bas[sh_id*BAS_SLOTS+ATOM_OF];
    double *expi = env + bas[sh_id*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[sh_id*BAS_SLOTS+PTR_COEFF];
    double *ri = env + atm[ia*ATM_SLOTS+PTR_COORD];
    for (int ip = 0; ip < iprim; ++ip) {
        __syncthreads();
        double ai = expi[ip];
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double kR = kx * xi + ky * yi + kz * zi;
        sincos(-kR, gzI, gzR);
        double Kab = exp(-.25/ai*kk);
        gxR[0] = OVERLAP_FAC * ci[ip] / (ai * sqrt(ai));
        gxI[0] = 0.;
        gyR[0] = 1.;
        gyI[0] = 0.;
        gzR[0] *= Kab;
        gzI[0] *= Kab;

        if (li > 0) {
            double a2 = .5 / ai;
            double xpaI = -a2 * kx;
            double ypaI = -a2 * ky;
            double zpaI = -a2 * kz;
            s0xR = gxR[0];
            s0xI = gxI[0];
            s0yR = gyR[0];
            s0yI = gyI[0];
            s0zR = gzR[0];
            s0zI = gzI[0];
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
            for (int i = 2; i <= AUXL; i++) {
                if (i > li) break;
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
        }
        __syncthreads();
#pragma unroll
        for (int n = 0; n < aux_nf; ++n) {
            if (n >= nfi) break;
            int addrx = idx[n*3+0] * NG_PER_BLOCK;
            int addry = idx[n*3+1] * NG_PER_BLOCK;
            int addrz = idx[n*3+2] * NG_PER_BLOCK;
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

    if (Gv_id < nGv) {
        int stride = nGv * OF_COMPLEX;
        double *aft_tensor = out + (envs.ao_loc[sh_id] * nGv + Gv_id) * OF_COMPLEX;
#pragma unroll
        for (int n = 0; n < aux_nf; ++n) {
            if (n >= nfi) break;
            aft_tensor[n*stride  ] = goutR[n];
            aft_tensor[n*stride+1] = goutI[n];
        }
    }
}

__global__ static
void ft_aopair_kernel(double *out, PBCIntEnvVars envs, double *pool, int *shl_pair_offsets,
                      uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                      int *gout_stride_lookup, int *ao_pair_loc, int ao_pair_offset,
                      double *Gv, int nGv, int *ao_loc, int compressing, int to_sph
                      #ifdef USE_SYCL
                      , sycl::nd_item<2> &item, char *shm_mem
                      #endif
                      )
{
    #ifdef USE_SYCL
    int sp_block_id = item.get_group_range(1) - item.get_group(1) - 1;
    int Gv_block_id = item.get_group(0);
    int Gv_id_in_block = item.get_local_id(1);
    int warp_id = item.get_local_id(0);
    int blockDim_y = item.get_local_range(0);

    auto thread_block = item.get_group();
    int &shl_pair0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &shl_pair1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &li = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &lj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &iprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jprim = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nao = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &gout_stride = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &nsp_per_block = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &img_max = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int (&img_counts)[WARPS] = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[WARPS]>(thread_block);

    double *shared_memory = reinterpret_cast<double*>(shm_mem);
    #else
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int Gv_block_id = blockIdx.y;
    int Gv_id_in_block = threadIdx.x;
    int warp_id = threadIdx.y;
    int blockDim_y = blockDim.y;

    __shared__ int shl_pair0, shl_pair1;
    __shared__ int li, lj;
    __shared__ int iprim, jprim;
    __shared__ int nao;
    __shared__ int gout_stride, nsp_per_block;
    __shared__ int img_max;
    __shared__ int img_counts[WARPS];

    extern __shared__ double shared_memory[];
    #endif

    constexpr int nGv_per_block = NG_PER_BLOCK;
    int thread_id = Gv_id_in_block + nGv_per_block * warp_id;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / bvk_nbas;
        int jsh0 = bas_ij0 - bvk_nbas * ish0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        // Note: must use this ao_loc than envs.ao_loc because envs.ao_loc
        // cannot handle spherical integrals
        nao = ao_loc[envs.nbas];
        gout_stride = gout_stride_lookup[li*LMAX1+lj];
        nsp_per_block = blockDim_y / gout_stride;
    }
    __syncthreads();
    int nGsp_per_block = nGv_per_block * nsp_per_block;
    int gout_id = warp_id / nsp_per_block;
    int sp_id = warp_id - nsp_per_block * gout_id;

    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    int stride_j = li + 1;
    int g_size = stride_j * (lj + 1);
    int gx_len = g_size * nGsp_per_block;
    double *gxR = shared_memory + nGv_per_block * sp_id + Gv_id_in_block;
    double *gxI = gxR + gx_len;
    double *gyR = gxR + gx_len*2;
    double *gyI = gxR + gx_len*3;
    double *gzR = gxR + gx_len*4;
    double *gzI = gxR + gx_len*5;
    double *rjri = shared_memory + gx_len*6 + sp_id;
    int *idx_i = (int*)(shared_memory + gx_len*6+nsp_per_block*3);
    int *idx_j = idx_i + nfi * 3;
    if (thread_id < nfi * 3) {
        idx_i[thread_id] = lex_xyz_address(li, thread_id) * nGsp_per_block;
        idx_i[thread_id] += (thread_id % 3) * gx_len * OF_COMPLEX;
    }
    if (thread_id < nfj * 3) {
        idx_j[thread_id] = lex_xyz_address(lj, thread_id) * stride_j * nGsp_per_block;
    }
    double *c2s_pool = pool + get_smid() * POOL_SIZE;

    int Gv_id = Gv_block_id * nGv_per_block + Gv_id_in_block;
    double kx = Gv[Gv_id];
    double ky = Gv[Gv_id+nGv];
    double kz = Gv[Gv_id+nGv * 2];
    double kk = kx * kx + ky * ky + kz * kz;

    for (int pair_idx = shl_pair0+sp_id; pair_idx < shl_pair1+sp_id; pair_idx += nsp_per_block) {
        __syncthreads();
        int pair_ij = pair_idx;
        if (pair_idx >= shl_pair1) {
            pair_ij = shl_pair0;
        }
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij % bvk_nbas;
        int img0 = img_offsets[pair_ij];
        int img1 = img_offsets[pair_ij+1];
        if (Gv_id_in_block == 0) {
            img_counts[warp_id] = img1 - img0;
        }
        __syncthreads();
        if (thread_id < WARPS) {
            int count = img_counts[thread_id];
            unsigned mask = (1u << WARPS) - 1;
            for (int offset = WARPS/2; offset > 0; offset /= 2) {
                count = max(count, __shfl_down_sync(mask, count, offset));
            }
            if (thread_id == 0) {
                img_max = count;
            }
        }
        __syncthreads();

        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xi = ri[0];
        double yi = ri[1];
        double zi = ri[2];
        double xjxi = rj[0] - xi;
        double yjyi = rj[1] - yi;
        double zjzi = rj[2] - zi;
        double goutR[GOUT_WIDTH];
        double goutI[GOUT_WIDTH];
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) {
            goutR[n] = 0.;
            goutI[n] = 0.;
        }
        double s0xR, s1xR, s2xR;
        double s0xI, s1xI, s2xI;
        int ijprim = iprim * jprim;
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
            for (int img = img0; img < img0+img_max; img++) {
                __syncthreads();
                int img_id = 0;
                if (img < img1) {
                    img_id = img_idx[img];
                } else {
                    fac = 0;
                }
                double Lx = img_coords[img_id*3+0];
                double Ly = img_coords[img_id*3+1];
                double Lz = img_coords[img_id*3+2];
                double xjLxi = xjxi + Lx;
                double yjLyi = yjyi + Ly;
                double zjLzi = zjzi + Lz;
                rjri[0*nsp_per_block] = xjLxi;
                rjri[1*nsp_per_block] = yjLyi;
                rjri[2*nsp_per_block] = zjLzi;
                if (gout_id == 0) {
                    double xij = xjLxi * aj_aij + xi;
                    double yij = yjLyi * aj_aij + yi;
                    double zij = zjLzi * aj_aij + zi;
                    double kR = kx * xij + ky * yij + kz * zij;
                    sincos(-kR, gzI, gzR);
                    double rr = xjLxi*xjLxi + yjLyi*yjLyi + zjLzi*zjLzi;
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
                int lij = li + lj;
                if (lij > 0) {
                    // gx[i+1] = ia2 * gx[i-1] + (rijrx[0] - kx[n]*a2*_Complex_I) * gx[i];
                    __syncthreads();
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gxR = gxR + n * gx_len * OF_COMPLEX;
                        double *_gxI = _gxR + gx_len;
                        double RpaR = rjri[n*nsp_per_block] * aj_aij; // Rp - Ra
                        double RpaI = -a2 * Gv[Gv_id+nGv*n];
                        s0xR = _gxR[0];
                        s0xI = _gxI[0];
                        s1xR = RpaR * s0xR - RpaI * s0xI;
                        s1xI = RpaR * s0xI + RpaI * s0xR;
                        _gxR[nGsp_per_block] = s1xR;
                        _gxI[nGsp_per_block] = s1xI;
                        for (int i = 1; i < lij; i++) {
                            double ia2 = i * a2;
                            s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
                            s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
                            _gxR[(i+1)*nGsp_per_block] = s2xR;
                            _gxI[(i+1)*nGsp_per_block] = s2xI;
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
                        double xjxi = rjri[_ix*nsp_per_block];
                        for (int j = 0; j < lj; ++j) {
                            int ij = (lij-j) + j*stride_j;
                            s1xR = _gx[ij*nGsp_per_block];
                            for (--ij; ij >= j*stride_j; --ij) {
                                s0xR = _gx[ij*nGsp_per_block];
                                _gx[(ij+stride_j)*nGsp_per_block] = s1xR - xjxi * s0xR;
                                s1xR = s0xR;
                            }
                        }
                    }
                }
                __syncthreads();
                if (pair_idx < shl_pair1 && img < img1) {
                    float div_nfi = c_div_nf[li];
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        uint32_t ij = n*gout_stride + gout_id;
                        if (ij >= nfij) break;
                        uint32_t j = ij * div_nfi;
                        uint32_t i = ij - nfi * j;
                        int addrx = idx_i[i*3+0] + idx_j[j*3+0];
                        int addry = idx_i[i*3+1] + idx_j[j*3+1];
                        int addrz = idx_i[i*3+2] + idx_j[j*3+2];
                        double xR = gxR[addrx];
                        double xI = gxR[addrx+gx_len];
                        double yR = gxR[addry];
                        double yI = gxR[addry+gx_len];
                        double zR = gxR[addrz];
                        double zI = gxR[addrz+gx_len];
                        double xyR = xR * yR - xI * yI;
                        double xyI = xR * yI + xI * yR;
                        goutR[n] += xyR * zR - xyI * zI;
                        goutI[n] += xyR * zI + xyI * zR;
                    }
                }
            }
        }

        size_t pair_offset;
        size_t bvk_Nao = ncells * nao;
        if (compressing) {
            pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        } else {
            int i0 = ao_loc[ish];
            int cell_id = jsh / envs.nbas;
            int jsh_cell0 = jsh - cell_id * envs.nbas;
            int j0 = ao_loc[jsh_cell0];
            size_t Nao = nao;
            pair_offset = (i0 * ncells + cell_id) * Nao + j0;
        }
        if (pair_idx < shl_pair1 && Gv_id < nGv) {
            if (to_sph && (li > 1 || lj > 1)) {
                double *out_local = c2s_pool +
                    (sp_id * nfij * nGv_per_block + Gv_id_in_block) * OF_COMPLEX;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ij = n*gout_stride + gout_id;
                    if (ij >= nfij) break;
                    size_t addr = ij * nGv_per_block * OF_COMPLEX;
                    out_local[addr  ] = goutR[n];
                    out_local[addr+1] = goutI[n];
                }
            } else {
                double *aft_tensor = out + (pair_offset * nGv + Gv_id) * OF_COMPLEX;
#pragma unroll
                for (int n = 0; n < GOUT_WIDTH; ++n) {
                    int ij = n*gout_stride + gout_id;
                    if (ij >= nfij) break;
                    size_t addr = ij;
                    if (!compressing) {
                        size_t j = ij / nfi;
                        size_t i = ij - nfi * j;
                        addr = i * bvk_Nao + j;
                    }
                    addr *= nGv * OF_COMPLEX;
                    aft_tensor[addr  ] = goutR[n];
                    aft_tensor[addr+1] = goutI[n];
                }
            }
        }
        __syncthreads();
        if (pair_idx < shl_pair1 && to_sph && (li > 1 || lj > 1)) {
            int di = li * 2 + 1;
            int nGv_c = nGv * OF_COMPLEX;
            int nGv_in_pool = nGv_per_block * OF_COMPLEX;
            size_t i_stride = nGv_c;
            size_t j_stride = nGv_c * di;
            if (!compressing) {
                i_stride = nGv_c * bvk_Nao;
                j_stride = nGv_c;
            }
            int Gv_start = Gv_block_id * nGv_per_block;
            double *inp_local = c2s_pool + sp_id * nfij * nGv_in_pool;
            double *aft_tensor = out + (pair_offset * nGv + Gv_start) * OF_COMPLEX;
            // Note each block within the compressed data in the input is transposed
            // for block with shape [nfi,nfj], i is accessed with smaller strides
            int comb_id = gout_id * nGv_per_block + Gv_id_in_block;
            int comb_stride = nGv_per_block * gout_stride;
            for (int k = comb_id; k < min(nGv_per_block, nGv-Gv_start)*OF_COMPLEX; k += comb_stride) {
                for (int j = 0; j < nfj; j++) {
                    double *inp = inp_local + j*nfi * nGv_in_pool + k;
                    for (int i = 0; i < di; i++) {
                        double *sph_out = aft_tensor + i * i_stride + k;
                        double s = 0;
                        // cart2sph for i
                        switch (li*li+i) {
                        case 0: { // l=0, m=0
                            s += inp[nGv_in_pool*0] * 1;
                        } break;
                        case 1: { // l=1, m=0
                            s += inp[nGv_in_pool*0] * 1;
                        } break;
                        case 2: { // l=1, m=1
                            s += inp[nGv_in_pool*1] * 1;
                        } break;
                        case 3: { // l=1, m=2
                            s += inp[nGv_in_pool*2] * 1;
                        } break;
                        case 4: { // l=2, m=0
                            s += inp[nGv_in_pool*1] * 1.092548430592079070;
                        } break;
                        case 5: { // l=2, m=1
                            s += inp[nGv_in_pool*4] * 1.092548430592079070;
                        } break;
                        case 6: { // l=2, m=2
                            s += inp[nGv_in_pool*0] * -0.315391565252520002;
                            s += inp[nGv_in_pool*3] * -0.315391565252520002;
                            s += inp[nGv_in_pool*5] * 0.630783130505040012;
                        } break;
                        case 7: { // l=2, m=3
                            s += inp[nGv_in_pool*2] * 1.092548430592079070;
                        } break;
                        case 8: { // l=2, m=4
                            s += inp[nGv_in_pool*0] * 0.546274215296039535;
                            s += inp[nGv_in_pool*3] * -0.546274215296039535;
                        } break;
                        case 9: { // l=3, m=0
                            s += inp[nGv_in_pool*1] * 1.770130769779930531;
                            s += inp[nGv_in_pool*6] * -0.590043589926643510;
                        } break;
                        case 10: { // l=3, m=1
                            s += inp[nGv_in_pool*4] * 2.890611442640554055;
                        } break;
                        case 11: { // l=3, m=2
                            s += inp[nGv_in_pool*1] * -0.457045799464465739;
                            s += inp[nGv_in_pool*6] * -0.457045799464465739;
                            s += inp[nGv_in_pool*8] * 1.828183197857862944;
                        } break;
                        case 12: { // l=3, m=3
                            s += inp[nGv_in_pool*2] * -1.119528997770346170;
                            s += inp[nGv_in_pool*7] * -1.119528997770346170;
                            s += inp[nGv_in_pool*9] * 0.746352665180230782;
                        } break;
                        case 13: { // l=3, m=4
                            s += inp[nGv_in_pool*0] * -0.457045799464465739;
                            s += inp[nGv_in_pool*3] * -0.457045799464465739;
                            s += inp[nGv_in_pool*5] * 1.828183197857862944;
                        } break;
                        case 14: { // l=3, m=5
                            s += inp[nGv_in_pool*2] * 1.445305721320277020;
                            s += inp[nGv_in_pool*7] * -1.445305721320277020;
                        } break;
                        case 15: { // l=3, m=6
                            s += inp[nGv_in_pool*0] * 0.590043589926643510;
                            s += inp[nGv_in_pool*3] * -1.770130769779930530;
                        } break;
                        case 16: { // l=4, m=0
                            s += inp[nGv_in_pool*1] * 2.503342941796704538;
                            s += inp[nGv_in_pool*6] * -2.503342941796704530;
                        } break;
                        case 17: { // l=4, m=1
                            s += inp[nGv_in_pool*4] * 5.310392309339791593;
                            s += inp[nGv_in_pool*11] * -1.770130769779930530;
                        } break;
                        case 18: { // l=4, m=2
                            s += inp[nGv_in_pool*1] * -0.946174695757560014;
                            s += inp[nGv_in_pool*6] * -0.946174695757560014;
                            s += inp[nGv_in_pool*8] * 5.677048174545360108;
                        } break;
                        case 19: { // l=4, m=3
                            s += inp[nGv_in_pool*4] * -2.007139630671867500;
                            s += inp[nGv_in_pool*11] * -2.007139630671867500;
                            s += inp[nGv_in_pool*13] * 2.676186174229156671;
                        } break;
                        case 20: { // l=4, m=4
                            s += inp[nGv_in_pool*0] * 0.317356640745612911;
                            s += inp[nGv_in_pool*3] * 0.634713281491225822;
                            s += inp[nGv_in_pool*5] * -2.538853125964903290;
                            s += inp[nGv_in_pool*10] * 0.317356640745612911;
                            s += inp[nGv_in_pool*12] * -2.538853125964903290;
                            s += inp[nGv_in_pool*14] * 0.846284375321634430;
                        } break;
                        case 21: { // l=4, m=5
                            s += inp[nGv_in_pool*2] * -2.007139630671867500;
                            s += inp[nGv_in_pool*7] * -2.007139630671867500;
                            s += inp[nGv_in_pool*9] * 2.676186174229156671;
                        } break;
                        case 22: { // l=4, m=6
                            s += inp[nGv_in_pool*0] * -0.473087347878780002;
                            s += inp[nGv_in_pool*5] * 2.838524087272680054;
                            s += inp[nGv_in_pool*10] * 0.473087347878780009;
                            s += inp[nGv_in_pool*12] * -2.838524087272680050;
                        } break;
                        case 23: { // l=4, m=7
                            s += inp[nGv_in_pool*2] * 1.770130769779930531;
                            s += inp[nGv_in_pool*7] * -5.310392309339791590;
                        } break;
                        case 24: { // l=4, m=8
                            s += inp[nGv_in_pool*0] * 0.625835735449176134;
                            s += inp[nGv_in_pool*3] * -3.755014412695056800;
                            s += inp[nGv_in_pool*10] * 0.625835735449176134;
                        } break;
                        case 25: { // l=5, m=0
                            s += inp[nGv_in_pool*1] * 3.281910284200850514;
                            s += inp[nGv_in_pool*6] * -6.563820568401701020;
                            s += inp[nGv_in_pool*15] * 0.656382056840170102;
                        } break;
                        case 26: { // l=5, m=1
                            s += inp[nGv_in_pool*4] * 8.302649259524165115;
                            s += inp[nGv_in_pool*11] * -8.302649259524165110;
                        } break;
                        case 27: { // l=5, m=2
                            s += inp[nGv_in_pool*1] * -1.467714898305751160;
                            s += inp[nGv_in_pool*6] * -0.978476598870500779;
                            s += inp[nGv_in_pool*8] * 11.741719186446009300;
                            s += inp[nGv_in_pool*15] * 0.489238299435250387;
                            s += inp[nGv_in_pool*17] * -3.913906395482003100;
                        } break;
                        case 28: { // l=5, m=3
                            s += inp[nGv_in_pool*4] * -4.793536784973323750;
                            s += inp[nGv_in_pool*11] * -4.793536784973323750;
                            s += inp[nGv_in_pool*13] * 9.587073569946647510;
                        } break;
                        case 29: { // l=5, m=4
                            s += inp[nGv_in_pool*1] * 0.452946651195696921;
                            s += inp[nGv_in_pool*6] * 0.905893302391393842;
                            s += inp[nGv_in_pool*8] * -5.435359814348363050;
                            s += inp[nGv_in_pool*15] * 0.452946651195696921;
                            s += inp[nGv_in_pool*17] * -5.435359814348363050;
                            s += inp[nGv_in_pool*19] * 3.623573209565575370;
                        } break;
                        case 30: { // l=5, m=5
                            s += inp[nGv_in_pool*2] * 1.754254836801353946;
                            s += inp[nGv_in_pool*7] * 3.508509673602707893;
                            s += inp[nGv_in_pool*9] * -4.678012898136943850;
                            s += inp[nGv_in_pool*16] * 1.754254836801353946;
                            s += inp[nGv_in_pool*18] * -4.678012898136943850;
                            s += inp[nGv_in_pool*20] * 0.935602579627388771;
                        } break;
                        case 31: { // l=5, m=6
                            s += inp[nGv_in_pool*0] * 0.452946651195696921;
                            s += inp[nGv_in_pool*3] * 0.905893302391393842;
                            s += inp[nGv_in_pool*5] * -5.435359814348363050;
                            s += inp[nGv_in_pool*10] * 0.452946651195696921;
                            s += inp[nGv_in_pool*12] * -5.435359814348363050;
                            s += inp[nGv_in_pool*14] * 3.623573209565575370;
                        } break;
                        case 32: { // l=5, m=7
                            s += inp[nGv_in_pool*2] * -2.396768392486661870;
                            s += inp[nGv_in_pool*9] * 4.793536784973323755;
                            s += inp[nGv_in_pool*16] * 2.396768392486661877;
                            s += inp[nGv_in_pool*18] * -4.793536784973323750;
                        } break;
                        case 33: { // l=5, m=8
                            s += inp[nGv_in_pool*0] * -0.489238299435250389;
                            s += inp[nGv_in_pool*3] * 0.978476598870500775;
                            s += inp[nGv_in_pool*5] * 3.913906395482003101;
                            s += inp[nGv_in_pool*10] * 1.467714898305751163;
                            s += inp[nGv_in_pool*12] * -11.741719186446009300;
                        } break;
                        case 34: { // l=5, m=9
                            s += inp[nGv_in_pool*2] * 2.075662314881041278;
                            s += inp[nGv_in_pool*7] * -12.453973889286247600;
                            s += inp[nGv_in_pool*16] * 2.075662314881041278;
                        } break;
                        case 35: { // l=5, m=10
                            s += inp[nGv_in_pool*0] * 0.656382056840170102;
                            s += inp[nGv_in_pool*3] * -6.563820568401701020;
                            s += inp[nGv_in_pool*10] * 3.281910284200850514;
                        } break;
                        case 36: { // l=6, m=0
                            s += inp[nGv_in_pool*1] * 4.0991046311514863;
                            s += inp[nGv_in_pool*6] * -13.6636821038382887;
                            s += inp[nGv_in_pool*15] * 4.0991046311514863;
                        } break;
                        case 37: { // l=6, m=1
                            s += inp[nGv_in_pool*4] * 11.8330958111587634;
                            s += inp[nGv_in_pool*11] * -23.6661916223175268;
                            s += inp[nGv_in_pool*22] * 2.3666191622317525;
                        } break;
                        case 38: { // l=6, m=2
                            s += inp[nGv_in_pool*1] * -2.0182596029148963;
                            s += inp[nGv_in_pool*8] * 20.1825960291489679;
                            s += inp[nGv_in_pool*15] * 2.0182596029148963;
                            s += inp[nGv_in_pool*17] * -20.1825960291489679;
                        } break;
                        case 39: { // l=6, m=3
                            s += inp[nGv_in_pool*4] * -8.2908473356343109;
                            s += inp[nGv_in_pool*11] * -5.5272315570895412;
                            s += inp[nGv_in_pool*13] * 22.1089262283581647;
                            s += inp[nGv_in_pool*22] * 2.7636157785447706;
                            s += inp[nGv_in_pool*24] * -7.3696420761193888;
                        } break;
                        case 40: { // l=6, m=4
                            s += inp[nGv_in_pool*1] * 0.9212052595149236;
                            s += inp[nGv_in_pool*6] * 1.8424105190298472;
                            s += inp[nGv_in_pool*8] * -14.7392841522387776;
                            s += inp[nGv_in_pool*15] * 0.9212052595149236;
                            s += inp[nGv_in_pool*17] * -14.7392841522387776;
                            s += inp[nGv_in_pool*19] * 14.7392841522387776;
                        } break;
                        case 41: { // l=6, m=5
                            s += inp[nGv_in_pool*4] * 2.9131068125936568;
                            s += inp[nGv_in_pool*11] * 5.8262136251873136;
                            s += inp[nGv_in_pool*13] * -11.6524272503746271;
                            s += inp[nGv_in_pool*22] * 2.9131068125936568;
                            s += inp[nGv_in_pool*24] * -11.6524272503746271;
                            s += inp[nGv_in_pool*26] * 4.6609709001498505;
                        } break;
                        case 42: { // l=6, m=6
                            s += inp[nGv_in_pool*0] * -0.3178460113381421;
                            s += inp[nGv_in_pool*3] * -0.9535380340144264;
                            s += inp[nGv_in_pool*5] * 5.7212282040865583;
                            s += inp[nGv_in_pool*10] * -0.9535380340144264;
                            s += inp[nGv_in_pool*12] * 11.4424564081731166;
                            s += inp[nGv_in_pool*14] * -7.6283042721154111;
                            s += inp[nGv_in_pool*21] * -0.3178460113381421;
                            s += inp[nGv_in_pool*23] * 5.7212282040865583;
                            s += inp[nGv_in_pool*25] * -7.6283042721154111;
                            s += inp[nGv_in_pool*27] * 1.0171072362820548;
                        } break;
                        case 43: { // l=6, m=7
                            s += inp[nGv_in_pool*2] * 2.9131068125936568;
                            s += inp[nGv_in_pool*7] * 5.8262136251873136;
                            s += inp[nGv_in_pool*9] * -11.6524272503746271;
                            s += inp[nGv_in_pool*16] * 2.9131068125936568;
                            s += inp[nGv_in_pool*18] * -11.6524272503746271;
                            s += inp[nGv_in_pool*20] * 4.6609709001498505;
                        } break;
                        case 44: { // l=6, m=8
                            s += inp[nGv_in_pool*0] * 0.4606026297574618;
                            s += inp[nGv_in_pool*3] * 0.4606026297574618;
                            s += inp[nGv_in_pool*5] * -7.3696420761193888;
                            s += inp[nGv_in_pool*10] * -0.4606026297574618;
                            s += inp[nGv_in_pool*14] * 7.3696420761193888;
                            s += inp[nGv_in_pool*21] * -0.4606026297574618;
                            s += inp[nGv_in_pool*23] * 7.3696420761193888;
                            s += inp[nGv_in_pool*25] * -7.3696420761193888;
                        } break;
                        case 45: { // l=6, m=9
                            s += inp[nGv_in_pool*2] * -2.7636157785447706;
                            s += inp[nGv_in_pool*7] * 5.5272315570895412;
                            s += inp[nGv_in_pool*9] * 7.3696420761193888;
                            s += inp[nGv_in_pool*16] * 8.2908473356343109;
                            s += inp[nGv_in_pool*18] * -22.1089262283581647;
                        } break;
                        case 46: { // l=6, m=10
                            s += inp[nGv_in_pool*0] * -0.5045649007287241;
                            s += inp[nGv_in_pool*3] * 2.5228245036436201;
                            s += inp[nGv_in_pool*5] * 5.0456490072872420;
                            s += inp[nGv_in_pool*10] * 2.5228245036436201;
                            s += inp[nGv_in_pool*12] * -30.2738940437234518;
                            s += inp[nGv_in_pool*21] * -0.5045649007287241;
                            s += inp[nGv_in_pool*23] * 5.0456490072872420;
                        } break;
                        case 47: { // l=6, m=11
                            s += inp[nGv_in_pool*2] * 2.3666191622317525;
                            s += inp[nGv_in_pool*7] * -23.6661916223175268;
                            s += inp[nGv_in_pool*16] * 11.8330958111587634;
                        } break;
                        case 48: { // l=6, m=12
                            s += inp[nGv_in_pool*0] * 0.6831841051919144;
                            s += inp[nGv_in_pool*3] * -10.2477615778787161;
                            s += inp[nGv_in_pool*10] * 10.2477615778787161;
                            s += inp[nGv_in_pool*21] * -0.6831841051919144;
                        } break;
                        }
                        // cart2sph for j
                        switch (j+nfj*lj/3) {
                        case 0: { // l=0, j=0
                            sph_out[0*j_stride] += s * 1;
                        } break;
                        case 1: { // l=1, j=0
                            sph_out[0*j_stride] += s * 1;
                        } break;
                        case 2: { // l=1, j=1
                            sph_out[1*j_stride] += s * 1;
                        } break;
                        case 3: { // l=1, j=2
                            sph_out[2*j_stride] += s * 1;
                        } break;
                        case 4: { // l=2, j=0
                            sph_out[2*j_stride] += s * -0.315391565252520002;
                            sph_out[4*j_stride] += s * 0.546274215296039535;
                        } break;
                        case 5: { // l=2, j=1
                            sph_out[0*j_stride] += s * 1.092548430592079070;
                        } break;
                        case 6: { // l=2, j=2
                            sph_out[3*j_stride] += s * 1.092548430592079070;
                        } break;
                        case 7: { // l=2, j=3
                            sph_out[2*j_stride] += s * -0.315391565252520002;
                            sph_out[4*j_stride] += s * -0.546274215296039535;
                        } break;
                        case 8: { // l=2, j=4
                            sph_out[1*j_stride] += s * 1.092548430592079070;
                        } break;
                        case 9: { // l=2, j=5
                            sph_out[2*j_stride] += s * 0.630783130505040012;
                        } break;
                        case 10: { // l=3, j=0
                            sph_out[4*j_stride] += s * -0.457045799464465739;
                            sph_out[6*j_stride] += s * 0.590043589926643510;
                        } break;
                        case 11: { // l=3, j=1
                            sph_out[0*j_stride] += s * 1.770130769779930531;
                            sph_out[2*j_stride] += s * -0.457045799464465739;
                        } break;
                        case 12: { // l=3, j=2
                            sph_out[3*j_stride] += s * -1.119528997770346170;
                            sph_out[5*j_stride] += s * 1.445305721320277020;
                        } break;
                        case 13: { // l=3, j=3
                            sph_out[4*j_stride] += s * -0.457045799464465739;
                            sph_out[6*j_stride] += s * -1.770130769779930530;
                        } break;
                        case 14: { // l=3, j=4
                            sph_out[1*j_stride] += s * 2.890611442640554055;
                        } break;
                        case 15: { // l=3, j=5
                            sph_out[4*j_stride] += s * 1.828183197857862944;
                        } break;
                        case 16: { // l=3, j=6
                            sph_out[0*j_stride] += s * -0.590043589926643510;
                            sph_out[2*j_stride] += s * -0.457045799464465739;
                        } break;
                        case 17: { // l=3, j=7
                            sph_out[3*j_stride] += s * -1.119528997770346170;
                            sph_out[5*j_stride] += s * -1.445305721320277020;
                        } break;
                        case 18: { // l=3, j=8
                            sph_out[2*j_stride] += s * 1.828183197857862944;
                        } break;
                        case 19: { // l=3, j=9
                            sph_out[3*j_stride] += s * 0.746352665180230782;
                        } break;
                        case 20: { // l=4, j=0
                            sph_out[4*j_stride] += s * 0.317356640745612911;
                            sph_out[6*j_stride] += s * -0.473087347878780002;
                            sph_out[8*j_stride] += s * 0.625835735449176134;
                        } break;
                        case 21: { // l=4, j=1
                            sph_out[0*j_stride] += s * 2.503342941796704538;
                            sph_out[2*j_stride] += s * -0.946174695757560014;
                        } break;
                        case 22: { // l=4, j=2
                            sph_out[5*j_stride] += s * -2.007139630671867500;
                            sph_out[7*j_stride] += s * 1.770130769779930531;
                        } break;
                        case 23: { // l=4, j=3
                            sph_out[4*j_stride] += s * 0.634713281491225822;
                            sph_out[8*j_stride] += s * -3.755014412695056800;
                        } break;
                        case 24: { // l=4, j=4
                            sph_out[1*j_stride] += s * 5.310392309339791593;
                            sph_out[3*j_stride] += s * -2.007139630671867500;
                        } break;
                        case 25: { // l=4, j=5
                            sph_out[4*j_stride] += s * -2.538853125964903290;
                            sph_out[6*j_stride] += s * 2.838524087272680054;
                        } break;
                        case 26: { // l=4, j=6
                            sph_out[0*j_stride] += s * -2.503342941796704530;
                            sph_out[2*j_stride] += s * -0.946174695757560014;
                        } break;
                        case 27: { // l=4, j=7
                            sph_out[5*j_stride] += s * -2.007139630671867500;
                            sph_out[7*j_stride] += s * -5.310392309339791590;
                        } break;
                        case 28: { // l=4, j=8
                            sph_out[2*j_stride] += s * 5.677048174545360108;
                        } break;
                        case 29: { // l=4, j=9
                            sph_out[5*j_stride] += s * 2.676186174229156671;
                        } break;
                        case 30: { // l=4, j=10
                            sph_out[4*j_stride] += s * 0.317356640745612911;
                            sph_out[6*j_stride] += s * 0.473087347878780009;
                            sph_out[8*j_stride] += s * 0.625835735449176134;
                        } break;
                        case 31: { // l=4, j=11
                            sph_out[1*j_stride] += s * -1.770130769779930530;
                            sph_out[3*j_stride] += s * -2.007139630671867500;
                        } break;
                        case 32: { // l=4, j=12
                            sph_out[4*j_stride] += s * -2.538853125964903290;
                            sph_out[6*j_stride] += s * -2.838524087272680050;
                        } break;
                        case 33: { // l=4, j=13
                            sph_out[3*j_stride] += s * 2.676186174229156671;
                        } break;
                        case 34: { // l=4, j=14
                            sph_out[4*j_stride] += s * 0.846284375321634430;
                        } break;
                        case 35: { // l=5, j=0
                            sph_out[6*j_stride] += s * 0.452946651195696921;
                            sph_out[8*j_stride] += s * -0.489238299435250389;
                            sph_out[10*j_stride] += s * 0.656382056840170102;
                        } break;
                        case 36: { // l=5, j=1
                            sph_out[0*j_stride] += s * 3.281910284200850514;
                            sph_out[2*j_stride] += s * -1.467714898305751160;
                            sph_out[4*j_stride] += s * 0.452946651195696921;
                        } break;
                        case 37: { // l=5, j=2
                            sph_out[5*j_stride] += s * 1.754254836801353946;
                            sph_out[7*j_stride] += s * -2.396768392486661870;
                            sph_out[9*j_stride] += s * 2.075662314881041278;
                        } break;
                        case 38: { // l=5, j=3
                            sph_out[6*j_stride] += s * 0.905893302391393842;
                            sph_out[8*j_stride] += s * 0.978476598870500775;
                            sph_out[10*j_stride] += s * -6.563820568401701020;
                        } break;
                        case 39: { // l=5, j=4
                            sph_out[1*j_stride] += s * 8.302649259524165115;
                            sph_out[3*j_stride] += s * -4.793536784973323750;
                        } break;
                        case 40: { // l=5, j=5
                            sph_out[6*j_stride] += s * -5.435359814348363050;
                            sph_out[8*j_stride] += s * 3.913906395482003101;
                        } break;
                        case 41: { // l=5, j=6
                            sph_out[0*j_stride] += s * -6.563820568401701020;
                            sph_out[2*j_stride] += s * -0.978476598870500779;
                            sph_out[4*j_stride] += s * 0.905893302391393842;
                        } break;
                        case 42: { // l=5, j=7
                            sph_out[5*j_stride] += s * 3.508509673602707893;
                            sph_out[9*j_stride] += s * -12.453973889286247600;
                        } break;
                        case 43: { // l=5, j=8
                            sph_out[2*j_stride] += s * 11.741719186446009300;
                            sph_out[4*j_stride] += s * -5.435359814348363050;
                        } break;
                        case 44: { // l=5, j=9
                            sph_out[5*j_stride] += s * -4.678012898136943850;
                            sph_out[7*j_stride] += s * 4.793536784973323755;
                        } break;
                        case 45: { // l=5, j=10
                            sph_out[6*j_stride] += s * 0.452946651195696921;
                            sph_out[8*j_stride] += s * 1.467714898305751163;
                            sph_out[10*j_stride] += s * 3.281910284200850514;
                        } break;
                        case 46: { // l=5, j=11
                            sph_out[1*j_stride] += s * -8.302649259524165110;
                            sph_out[3*j_stride] += s * -4.793536784973323750;
                        } break;
                        case 47: { // l=5, j=12
                            sph_out[6*j_stride] += s * -5.435359814348363050;
                            sph_out[8*j_stride] += s * -11.741719186446009300;
                        } break;
                        case 48: { // l=5, j=13
                            sph_out[3*j_stride] += s * 9.587073569946647510;
                        } break;
                        case 49: { // l=5, j=14
                            sph_out[6*j_stride] += s * 3.623573209565575370;
                        } break;
                        case 50: { // l=5, j=15
                            sph_out[0*j_stride] += s * 0.656382056840170102;
                            sph_out[2*j_stride] += s * 0.489238299435250387;
                            sph_out[4*j_stride] += s * 0.452946651195696921;
                        } break;
                        case 51: { // l=5, j=16
                            sph_out[5*j_stride] += s * 1.754254836801353946;
                            sph_out[7*j_stride] += s * 2.396768392486661877;
                            sph_out[9*j_stride] += s * 2.075662314881041278;
                        } break;
                        case 52: { // l=5, j=17
                            sph_out[2*j_stride] += s * -3.913906395482003100;
                            sph_out[4*j_stride] += s * -5.435359814348363050;
                        } break;
                        case 53: { // l=5, j=18
                            sph_out[5*j_stride] += s * -4.678012898136943850;
                            sph_out[7*j_stride] += s * -4.793536784973323750;
                        } break;
                        case 54: { // l=5, j=19
                            sph_out[4*j_stride] += s * 3.623573209565575370;
                        } break;
                        case 55: { // l=5, j=20
                            sph_out[5*j_stride] += s * 0.935602579627388771;
                        } break;
                        case 56: { // l=6, j=0
                            sph_out[6*j_stride] += s * -0.3178460113381421;
                            sph_out[8*j_stride] += s * 0.4606026297574618;
                            sph_out[10*j_stride] += s * -0.5045649007287241;
                            sph_out[12*j_stride] += s * 0.6831841051919144;
                        } break;
                        case 57: { // l=6, j=1
                            sph_out[0*j_stride] += s * 4.0991046311514863;
                            sph_out[2*j_stride] += s * -2.0182596029148963;
                            sph_out[4*j_stride] += s * 0.9212052595149236;
                        } break;
                        case 58: { // l=6, j=2
                            sph_out[7*j_stride] += s * 2.9131068125936568;
                            sph_out[9*j_stride] += s * -2.7636157785447706;
                            sph_out[11*j_stride] += s * 2.3666191622317525;
                        } break;
                        case 59: { // l=6, j=3
                            sph_out[6*j_stride] += s * -0.9535380340144264;
                            sph_out[8*j_stride] += s * 0.4606026297574618;
                            sph_out[10*j_stride] += s * 2.5228245036436201;
                            sph_out[12*j_stride] += s * -10.2477615778787161;
                        } break;
                        case 60: { // l=6, j=4
                            sph_out[1*j_stride] += s * 11.8330958111587634;
                            sph_out[3*j_stride] += s * -8.2908473356343109;
                            sph_out[5*j_stride] += s * 2.9131068125936568;
                        } break;
                        case 61: { // l=6, j=5
                            sph_out[6*j_stride] += s * 5.7212282040865583;
                            sph_out[8*j_stride] += s * -7.3696420761193888;
                            sph_out[10*j_stride] += s * 5.0456490072872420;
                        } break;
                        case 62: { // l=6, j=6
                            sph_out[0*j_stride] += s * -13.6636821038382887;
                            sph_out[4*j_stride] += s * 1.8424105190298472;
                        } break;
                        case 63: { // l=6, j=7
                            sph_out[7*j_stride] += s * 5.8262136251873136;
                            sph_out[9*j_stride] += s * 5.5272315570895412;
                            sph_out[11*j_stride] += s * -23.6661916223175268;
                        } break;
                        case 64: { // l=6, j=8
                            sph_out[2*j_stride] += s * 20.1825960291489679;
                            sph_out[4*j_stride] += s * -14.7392841522387776;
                        } break;
                        case 65: { // l=6, j=9
                            sph_out[7*j_stride] += s * -11.6524272503746271;
                            sph_out[9*j_stride] += s * 7.3696420761193888;
                        } break;
                        case 66: { // l=6, j=10
                            sph_out[6*j_stride] += s * -0.9535380340144264;
                            sph_out[8*j_stride] += s * -0.4606026297574618;
                            sph_out[10*j_stride] += s * 2.5228245036436201;
                            sph_out[12*j_stride] += s * 10.2477615778787161;
                        } break;
                        case 67: { // l=6, j=11
                            sph_out[1*j_stride] += s * -23.6661916223175268;
                            sph_out[3*j_stride] += s * -5.5272315570895412;
                            sph_out[5*j_stride] += s * 5.8262136251873136;
                        } break;
                        case 68: { // l=6, j=12
                            sph_out[6*j_stride] += s * 11.4424564081731166;
                            sph_out[10*j_stride] += s * -30.2738940437234518;
                        } break;
                        case 69: { // l=6, j=13
                            sph_out[3*j_stride] += s * 22.1089262283581647;
                            sph_out[5*j_stride] += s * -11.6524272503746271;
                        } break;
                        case 70: { // l=6, j=14
                            sph_out[6*j_stride] += s * -7.6283042721154111;
                            sph_out[8*j_stride] += s * 7.3696420761193888;
                        } break;
                        case 71: { // l=6, j=15
                            sph_out[0*j_stride] += s * 4.0991046311514863;
                            sph_out[2*j_stride] += s * 2.0182596029148963;
                            sph_out[4*j_stride] += s * 0.9212052595149236;
                        } break;
                        case 72: { // l=6, j=16
                            sph_out[7*j_stride] += s * 2.9131068125936568;
                            sph_out[9*j_stride] += s * 8.2908473356343109;
                            sph_out[11*j_stride] += s * 11.8330958111587634;
                        } break;
                        case 73: { // l=6, j=17
                            sph_out[2*j_stride] += s * -20.1825960291489679;
                            sph_out[4*j_stride] += s * -14.7392841522387776;
                        } break;
                        case 74: { // l=6, j=18
                            sph_out[7*j_stride] += s * -11.6524272503746271;
                            sph_out[9*j_stride] += s * -22.1089262283581647;
                        } break;
                        case 75: { // l=6, j=19
                            sph_out[4*j_stride] += s * 14.7392841522387776;
                        } break;
                        case 76: { // l=6, j=20
                            sph_out[7*j_stride] += s * 4.6609709001498505;
                        } break;
                        case 77: { // l=6, j=21
                            sph_out[6*j_stride] += s * -0.3178460113381421;
                            sph_out[8*j_stride] += s * -0.4606026297574618;
                            sph_out[10*j_stride] += s * -0.5045649007287241;
                            sph_out[12*j_stride] += s * -0.6831841051919144;
                        } break;
                        case 78: { // l=6, j=22
                            sph_out[1*j_stride] += s * 2.3666191622317525;
                            sph_out[3*j_stride] += s * 2.7636157785447706;
                            sph_out[5*j_stride] += s * 2.9131068125936568;
                        } break;
                        case 79: { // l=6, j=23
                            sph_out[6*j_stride] += s * 5.7212282040865583;
                            sph_out[8*j_stride] += s * 7.3696420761193888;
                            sph_out[10*j_stride] += s * 5.0456490072872420;
                        } break;
                        case 80: { // l=6, j=24
                            sph_out[3*j_stride] += s * -7.3696420761193888;
                            sph_out[5*j_stride] += s * -11.6524272503746271;
                        } break;
                        case 81: { // l=6, j=25
                            sph_out[6*j_stride] += s * -7.6283042721154111;
                            sph_out[8*j_stride] += s * -7.3696420761193888;
                        } break;
                        case 82: { // l=6, j=26
                            sph_out[5*j_stride] += s * 4.6609709001498505;
                        } break;
                        case 83: { // l=6, j=27
                            sph_out[6*j_stride] += s * 1.0171072362820548;
                        } break;
                        }
                    }
                }
            }
        }
    }
}

extern "C" {
int build_ft_ao(double *out, RysIntEnvVars *envs, int ngrids, double *grids, int nbas)
{
    int nsh_per_block = FT_AO_THREADS/NG_PER_BLOCK;
    int nbatches_grids = (ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK;
    int nbatches_shls = (nbas + nsh_per_block - 1) / nsh_per_block;
    #ifdef USE_SYCL
    sycl::range<2> threads(nsh_per_block, NG_PER_BLOCK);
    sycl::range<2> blocks(nbatches_grids, nbatches_shls);
    auto dev_envs = *envs;
    sycl_get_queue()->parallel_for<class ft_ao_bdiv_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
      ft_ao_bdiv_kernel(out, dev_envs, ngrids, grids);
    });
    #else
    dim3 threads(NG_PER_BLOCK, nsh_per_block);
    dim3 blocks(nbatches_shls, nbatches_grids);
    ft_ao_bdiv_kernel<<<blocks, threads>>>(out, *envs, ngrids, grids);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_ao_bdiv_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}

int build_ft_aopair(double *out, PBCIntEnvVars *envs, double *pool,
                    int shm_size, int nbatches_shl_pair, int *shl_pair_offsets,
                    uint32_t *bas_ij_idx, int *img_idx, uint32_t *img_offsets,
                    int *gout_stride_lookup, int *ao_pair_loc, int ao_pair_offset,
                    double *grids, int ngrids, int *ao_loc, int compressing, int to_sph)
{
    constexpr int nGv_per_block = NG_PER_BLOCK;
    int Gv_batches = (ngrids + nGv_per_block - 1) / nGv_per_block;
    #ifdef USE_SYCL
    sycl::range<2> threads(WARPS, nGv_per_block);
    sycl::range<2> blocks(Gv_batches, nbatches_shl_pair);
    auto dev_envs = *envs;
    sycl_get_queue()->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(shm_size), cgh);
      cgh.parallel_for<class ft_aopair_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) {
        ft_aopair_kernel(out, dev_envs, pool, shl_pair_offsets, bas_ij_idx, img_idx, img_offsets,
                         gout_stride_lookup, ao_pair_loc, ao_pair_offset, grids, ngrids,
                         ao_loc, compressing, to_sph,
                         item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc));
      });
    });
    #else
    cudaFuncSetAttribute(ft_aopair_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 threads(nGv_per_block, WARPS);
    dim3 blocks(nbatches_shl_pair, Gv_batches);
    ft_aopair_kernel<<<blocks, threads, shm_size>>>(
        out, *envs, pool, shl_pair_offsets, bas_ij_idx, img_idx, img_offsets,
        gout_stride_lookup, ao_pair_loc, ao_pair_offset, grids, ngrids,
        ao_loc, compressing, to_sph);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ft_aopair_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    #endif
    return 0;
}
}
