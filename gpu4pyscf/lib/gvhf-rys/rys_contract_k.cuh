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

#ifdef USE_SYCL
#include "gint/sycl_device.hpp"

// Please mind that this is a copy of the values in rys_constant.cu
// Given the support for C++ seperate constexpr declaration and definition
// is until C++20, we do use this var as `inline constexpr`
static constexpr int _c_cartesian_lexical_xyz[117] = {
    // s, offset = 0
    0, 0, 0,
    0, 0, 0, // padding
    0, 0, 0, // padding
    // p, offset = 9
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    // d, offset = 9 * 2
    2, 0, 0,
    1, 1, 0,
    1, 0, 1,
    0, 2, 0,
    0, 1, 1,
    0, 0, 2,
    // f, offset = 9 * 4
    3, 0, 0,
    2, 1, 0,
    2, 0, 1,
    1, 2, 0,
    1, 1, 1,
    1, 0, 2,
    0, 3, 0,
    0, 2, 1,
    0, 1, 2,
    0, 0, 3,
    0, 0, 0, // padding
    0, 0, 0, // padding
    // g, offset = 9 * 8
    4, 0, 0,
    3, 1, 0,
    3, 0, 1,
    2, 2, 0,
    2, 1, 1,
    2, 0, 2,
    1, 3, 0,
    1, 2, 1,
    1, 1, 2,
    1, 0, 3,
    0, 4, 0,
    0, 3, 1,
    0, 2, 2,
    0, 1, 3,
    0, 0, 4,
};

#else
#include <cuda.h>
#include <cuda_runtime.h>

extern __constant__ int _c_cartesian_lexical_xyz[];

#endif


#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
int lex_xyz_offset(int l) {
    // the offsets for _c_cartesian_lexical_xyz are: 0, 1, 2, 4, 8, 13, 20, ...
    int offset = (1 << l) >> 1;
    return offset * 9;
}

#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
int lex_xyz_address(int l, int i)
{
    // the offsets for _c_cartesian_lexical_xyz are: 0, 1, 2, 4, 8, 13, 20, ...
    return _c_cartesian_lexical_xyz[lex_xyz_offset(l) + i];
}

template <int LIJ>
#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
void vrr(double *g, double *ri, double *rj, double *Rpq, double aj_aij, double rt_aij,
         double b10, int g_size)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int nsq_per_block = item.get_local_range(1);
    int gout_id = item.get_local_id(0);
    int gout_stride = item.get_local_range(0);
#else
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
#endif
    for (int n = gout_id; n < 3; n += gout_stride) {
        double *_gx = g + n * g_size * nsq_per_block;
        double Rpa = (rj[n] - ri[n]) * aj_aij;
        double c0x = Rpa - rt_aij * Rpq[n*nsq_per_block];
        double s0x, s1x, s2x;
        s0x = _gx[0];
        s1x = c0x * s0x;
        _gx[nsq_per_block] = s1x;
        for (int i = 1; i < LIJ; ++i) {
            s2x = c0x * s1x + i * b10 * s0x;
            _gx[(i+1)*nsq_per_block] = s2x;
            s0x = s1x;
            s1x = s2x;
        }
    }
}

template <int LKL>
#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
void trr(double *g, double *rlrk, double *Rpq, double al_akl, double rt_akl,
         double b00, double b01, int lij3, int stride_k, int g_size)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int nsq_per_block = item.get_local_range(1);
    int gout_id = item.get_local_id(0);
    int gout_stride = item.get_local_range(0);
#else
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
#endif
    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
        __syncthreads();
        int i = n / 3; //for i in range(lij+1):
        int _ix = n % 3; // TODO: remove _ix for nroots > 2
        double *_gx = g + (i + _ix * g_size) * nsq_per_block;
        double Rqc = rlrk[_ix*nsq_per_block] * al_akl;
        double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
        //for i in range(lij+1):
        //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
        double s0x, s1x, s2x;
        if (n < lij3) {
            s0x = _gx[0];
            s1x = cpx * s0x;
            s1x += i * b00 * _gx[-nsq_per_block];
            _gx[stride_k*nsq_per_block] = s1x;
        }

        //for k in range(1, lkl):
        //    for i in range(lij+1):
        //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
        for (int k = 1; k < LKL; ++k) {
            __syncthreads();
            if (n < lij3) {
                s2x = cpx*s1x + k*b01*s0x;
                s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block];
                _gx[(k*stride_k+stride_k)*nsq_per_block] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
        }
    }
}

template <int LI, int LJ>
#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
void hrr_ij(double *g, double *rjri, int count, int g_size)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int nsq_per_block = item.get_local_range(1);
    int gout_id = item.get_local_id(0);
    int gout_stride = item.get_local_range(0);
#else
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
#endif
    constexpr int lij = LI + LJ;
    constexpr int stride_j = LI + 1;
    constexpr int stride_k = stride_j * (LJ + 1);
    for (int m = gout_id; m < count; m += gout_stride) {
        int k = m / 3;
        int _ix = m % 3;
        double xjxi = rjri[_ix];
        double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
#pragma unroll
        for (int j = 0; j < LJ; ++j) {
            int ij = (lij-j) + j*stride_j;
            double s0x, s1x;
            s1x = _gx[ij*nsq_per_block];
#pragma unroll
            for (--ij; ij >= j*stride_j; --ij) {
                s0x = _gx[ij*nsq_per_block];
                _gx[(ij+stride_j)*nsq_per_block] = s1x - xjxi * s0x;
                s1x = s0x;
            }
        }
    }
}
template <int LK, int LL>
#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
void hrr_kl(double *g, double *rlrk, int stride_k)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int nsq_per_block = item.get_local_range(1);
    int gout_id = item.get_local_id(0);
    int gout_stride = item.get_local_range(0);
#else
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
#endif
    constexpr int lkl = LK + LL;
    for (int n = gout_id; n < stride_k*3; n += gout_stride) {
        int i = n / 3;
        int _ix = n % 3;
        double xlxk = rlrk[_ix*nsq_per_block];
        double *_gx = g + (_ix*stride_k*(LK+1)*(LL+1) + i) * nsq_per_block;
#pragma unroll
        for (int l = 0; l < LL; ++l) {
            int kl = (lkl+l*LK)*stride_k;
            double s0x, s1x;
            s1x = _gx[kl*nsq_per_block];
#pragma unroll
            for (int k = lkl-1; k >= l; --k) {
                int kl = (k+l*LK) * stride_k;
                s0x = _gx[kl*nsq_per_block];
                _gx[(kl+stride_k*(LK+1))*nsq_per_block] = s1x - xlxk * s0x;
                s1x = s0x;
            }
        }
    }
}

template <int I, int J, int K, int L>
#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
void inner_dot(double *gout, double *g,
               int *addr_i, int *addr_j,
               int *addr_k, int *addr_l)
{
#pragma unroll
    for (int l = 0; l < L; ++l) {
        int lx_addr = addr_l[l*3+0];
        int ly_addr = addr_l[l*3+1];
        int lz_addr = addr_l[l*3+2];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        int kx_addr = addr_k[k*3+0] + lx_addr;
        int ky_addr = addr_k[k*3+1] + ly_addr;
        int kz_addr = addr_k[k*3+2] + lz_addr;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        int jx_addr = addr_j[j*3+0] + kx_addr;;
        int jy_addr = addr_j[j*3+1] + ky_addr;;
        int jz_addr = addr_j[j*3+2] + kz_addr;;
#pragma unroll
    for (int i = 0; i < I; ++i) {
        int n = i + j * 3 + k * 9 + l * 27;
        int addrx = addr_i[i*3+0] + jx_addr;
        int addry = addr_i[i*3+1] + jy_addr;
        int addrz = addr_i[i*3+2] + jz_addr;
        gout[n] += g[addrx] * g[addry] * g[addrz];
    } } } }
}

#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
void load_dm(double *dm, double *dm_cache, int nao, int nfi, int nfj,
             int ldi, int ldj)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int t_id = item.get_local_id(0);
    int t_stride = item.get_local_range(0);
    int nsq_per_block = item.get_local_range(1);
#else
    int t_id = threadIdx.y;
    int t_stride = blockDim.y;
    int nsq_per_block = blockDim.x;
#endif
    for (int m = t_id; m < ldi*ldj; m += t_stride) {
        int i = m / ldj;
        int j = m % ldj;
        if (i < nfi && j < nfj) {
            dm_cache[m*nsq_per_block] = dm[i*nao+j];
        } else {
            dm_cache[m*nsq_per_block] = 0;
        }
    }
}

template <int I, int J, int K, int L>
#ifdef USE_SYCL
static inline
#else
__device__ __forceinline__
#endif
void dot_dm(double *vk, double *dm, double *gout, int nao, int i0, int l0,
            int ioff, int joff, int koff, int loff, int ldk, int nfi, int nfl, int active)
{
#ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    __syncthreads();    
    int nsq_per_block = item.get_local_range(1);
#else
    __syncthreads();    
    int nsq_per_block = blockDim.x;
#endif
    if (active) {
        int dl = nfl - loff;
        int di = nfi - ioff;
        double *dm_local = dm + (joff*ldk+koff)*nsq_per_block;
        double *vk_local = vk + (i0+ioff)*nao+(l0+loff);
#pragma unroll
        for (int l = 0; l < 3; ++l) {
            if (l >= dl) break;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (i >= di) break;
            double v = 0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                int n = i * I + j * J + k * K + l * L;
                v += gout[n] * dm_local[(j*ldk+k)*nsq_per_block];
            } }
            atomicAdd(vk_local+i*nao+l, v);
        } }
    }
    __syncthreads();
}
