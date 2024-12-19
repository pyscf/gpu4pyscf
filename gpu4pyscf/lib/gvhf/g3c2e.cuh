/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "gint/g2e.h"
#include "gint/cint2e.cuh"
#include "gvhf.h"
/*
__device__
static void GINTkernel_int3c2e_ip1_getjk(JKMatrix jk, double* __restrict__ gout,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int nao = jk.nao;
    int i, j, k, n, off_dm, off_rhok;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;
    double j3[GPU_CART_MAX * 3];
    double k3[GPU_CART_MAX * 3];

    for (i = 0; i < (i1-i0) * 3; i++){
        j3[i] = 0.0;
        k3[i] = 0.0;
    }
    for (n = 0, k = k0; k < k1; ++k) {
        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
                off_dm = i + nao*j;
                off_rhok = i + nao*j + k*nao*nao;

                double sx = gout[3*n];
                double sy = gout[3*n + 1];
                double sz = gout[3*n + 2];

                double rhoj_tmp = dm[off_dm] * rhoj[k];
                double rhok_tmp = rhok[off_rhok];

                int ii = 3*(i-i0);
                j3[ii + 0] += rhoj_tmp * sx;
                j3[ii + 1] += rhoj_tmp * sy;
                j3[ii + 2] += rhoj_tmp * sz;

                k3[ii + 0] += rhok_tmp * sx;
                k3[ii + 1] += rhok_tmp * sy;
                k3[ii + 2] += rhok_tmp * sz;
            }
        }
    }

    for (i = i0; i < i1; ++i){
        int ii = 3*(i-i0);
        atomicAdd(vj + i + 0*nao, j3[ii + 0]);
        atomicAdd(vj + i + 1*nao, j3[ii + 1]);
        atomicAdd(vj + i + 2*nao, j3[ii + 2]);

        atomicAdd(vk + i + 0*nao, k3[ii + 0]);
        atomicAdd(vk + i + 1*nao, k3[ii + 1]);
        atomicAdd(vk + i + 2*nao, k3[ii + 2]);
    }
}

__device__
static void GINTkernel_int3c2e_ip2_getjk(JKMatrix jk, double* __restrict__ gout,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int nao = jk.nao;
    int naux = jk.naux;
    int i, j, k, n, off_dm, off_rhok;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;
    double j3[GPU_CART_MAX * 3];
    double k3[GPU_CART_MAX * 3];

    for (k = 0; k < (k1-k0) * 3; k++){
        j3[k] = 0.0;
        k3[k] = 0.0;
    }

    for (n = 0, k = k0; k < k1; ++k) {
        for (j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
                off_dm = i + nao*j;
                off_rhok = i + nao*j + k*nao*nao;

                double sx = gout[3 * n];
                double sy = gout[3 * n + 1];
                double sz = gout[3 * n + 2];

                double rhoj_tmp = dm[off_dm] * rhoj[k];
                double rhok_tmp = rhok[off_rhok];

                int kk = 3*(k-k0);
                j3[kk + 0] += sx * rhoj_tmp;
                j3[kk + 1] += sy * rhoj_tmp;
                j3[kk + 2] += sz * rhoj_tmp;

                k3[kk + 0] += sx * rhok_tmp;
                k3[kk + 1] += sy * rhok_tmp;
                k3[kk + 2] += sz * rhok_tmp;
            }
        }
    }
    for (k = k0; k < k1; ++k){
        int kk = 3*(k-k0);

        atomicAdd(vj + k + 0*naux, j3[kk + 0]);
        atomicAdd(vj + k + 1*naux, j3[kk + 1]);
        atomicAdd(vj + k + 2*naux, j3[kk + 2]);

        atomicAdd(vk + k + 0*naux, k3[kk + 0]);
        atomicAdd(vk + k + 1*naux, k3[kk + 1]);
        atomicAdd(vk + k + 2*naux, k3[kk + 2]);
    }
}
*/
// jaux = numpy.einsum('ijk,ji->k', j3c, dm)
template <int NROOTS> __device__
static void GINTkernel_int3c2e_getj_pass1(GINTEnvVars envs, JKMatrix jk, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    int nao = jk.nao;
    int i, j, k;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ dm = jk.dm;

    int i_l = envs.i_l;
    int j_l = envs.j_l;
    int k_l = envs.k_l;
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    for (k = k0; k < k1; ++k) {
        int kp = k - k0;
        double rhoj_tmp = 0.0;
        for (j = j0; j < j1; ++j) {
            int jp = j - j0;
            for (i = i0; i < i1; ++i) {
                int ip = i - i0;

                int loc_k = c_l_locs[k_l] + kp;
                int loc_j = c_l_locs[j_l] + jp;
                int loc_i = c_l_locs[i_l] + ip;

                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + envs.g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + envs.g_size * 2;
                double s = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    s += g[ix+ir] * g[iy+ir] * g[iz+ir];
                }
                int off_dm = i + nao*j;
                rhoj_tmp += dm[off_dm] * s;
            }
        }
        atomicAdd(rhoj+k, rhoj_tmp);
    }
}

// vj = numpy.einsum('ijk,k->ij', j3c, rho)
template <int NROOTS> __device__
static void GINTkernel_int3c2e_getj_pass2(GINTEnvVars envs, JKMatrix jk, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    int nao = jk.nao;
    int i, j, k;
    double* __restrict__ vj = jk.vj;

    int i_l = envs.i_l;
    int j_l = envs.j_l;
    int k_l = envs.k_l;
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    double rhoj[GPU_CART_MAX];
    for (k = 0; k < k1-k0; k++){
        rhoj[k] = jk.rhoj[k0+k];
    }

    for (j = j0; j < j1; ++j) {
        int jp = j - j0;
        for (i = i0; i < i1; ++i) {
            int ip = i - i0;
            double vj_tmp = 0.0;
            for (k = k0; k < k1; ++k){
                int kp = k - k0;

                int loc_k = c_l_locs[k_l] + kp;
                int loc_j = c_l_locs[j_l] + jp;
                int loc_i = c_l_locs[i_l] + ip;

                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + envs.g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + envs.g_size * 2;
                double s = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    s += g[ix+ir] * g[iy+ir] * g[iz+ir];
                }
                vj_tmp += rhoj[k-k0] * s;
            }
            atomicAdd(vj+j+nao*i, vj_tmp);
        }
    }
}

/*
ij,  k,   nijk  -> ni
dm, rhoj, int3c_ip2 -> vj

ijk,     nijk   -> ni
rhok, int3c_ip2 -> vk
*/

template <int NROOTS> __device__
static void GINTkernel_int3c2e_ip1_getjk_direct(GINTEnvVars envs, JKMatrix jk, double* j3, double* k3, double* f, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    int g_size = envs.g_size;
    int nao = jk.nao;
    int i, j, k;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;

    int i_l = envs.i_l;
    int j_l = envs.j_l;
    int k_l = envs.k_l;
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    if (rhoj == NULL){
        for (k = k0; k < k1; ++k) {
            for (j = j0; j < j1; ++j) {
                for (i = i0; i < i1; ++i) {
                    int kp = k - k0;
                    int jp = j - j0;
                    int ip = i - i0;

                    int loc_k = c_l_locs[k_l] + kp;
                    int loc_j = c_l_locs[j_l] + jp;
                    int loc_i = c_l_locs[i_l] + ip;

                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }

                    int ii = 3*(i-i0);
                    int off_rhok = i + nao*j + k*nao*nao;
                    double rhok_tmp = rhok[off_rhok];
                    k3[ii + 0] += rhok_tmp * sx;
                    k3[ii + 1] += rhok_tmp * sy;
                    k3[ii + 2] += rhok_tmp * sz;
                }
            }
        }
        return;
    }

    if (rhok == NULL){
        for (i = i0; i < i1; ++i) {
            for (j = j0; j < j1; ++j) {
                double jx = 0.0;
                double jy = 0.0;
                double jz = 0.0;
                for (k = k0; k < k1; ++k) {
                    int ip = i - i0;
                    int jp = j - j0;
                    int kp = k - k0;
                    
                    int loc_j = c_l_locs[j_l] + jp;
                    int loc_i = c_l_locs[i_l] + ip;
                    int loc_k = c_l_locs[k_l] + kp;
                    
                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }

                    double rhoj_k = rhoj[k];
                    jx += rhoj_k * sx;
                    jy += rhoj_k * sy;
                    jz += rhoj_k * sz;
                }
                int ii = 3*(i-i0);
                int off_dm = i + nao*j;
                //double rhoj_tmp = dm[off_dm] * rhoj_k;
                double dm_ij = dm[off_dm];
                j3[ii + 0] += jx * dm_ij;
                j3[ii + 1] += jy * dm_ij;
                j3[ii + 2] += jz * dm_ij;
            }
        }
        return;
    }

    for (i = i0; i < i1; ++i) {
        for (j = j0; j < j1; ++j) {
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            for (k = k0; k < k1; ++k) {
                int jp = j - j0;
                int kp = k - k0;
                int ip = i - i0;
                
                int loc_k = c_l_locs[k_l] + kp;
                int loc_j = c_l_locs[j_l] + jp;
                int loc_i = c_l_locs[i_l] + ip;
                
                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    double gx = g[ix+ir];
                    double gy = g[iy+ir];
                    double gz = g[iz+ir];
                    sx += f[ix + ir] * gy * gz;
                    sy += gx * f[iy + ir] * gz;
                    sz += gx * gy * f[iz + ir];
                }

                int off_rhok = i + nao*j + k*nao*nao;
                double rhok_tmp = rhok[off_rhok];
                int ii = 3*(i-i0);
                k3[ii + 0] += rhok_tmp * sx;
                k3[ii + 1] += rhok_tmp * sy;
                k3[ii + 2] += rhok_tmp * sz;

                double rhoj_k = rhoj[k];
                jx += rhoj_k * sx;
                jy += rhoj_k * sy;
                jz += rhoj_k * sz;
            }
            int off_dm = i + nao*j;
            //double rhoj_tmp = dm[off_dm] * rhoj_k;
            double dm_ij = dm[off_dm];
            int ii = 3*(i-i0);
            
            j3[ii + 0] += jx * dm_ij;
            j3[ii + 1] += jy * dm_ij;
            j3[ii + 2] += jz * dm_ij;
        }
    }
}

template <int LI, int LJ, int LK> __device__
static void GINTkernel_int3c2e_ip1_getjk_direct(GINTEnvVars envs, JKMatrix jk, double* j3, double* k3, double* f, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;

    constexpr int LI_CEIL = LI + 1;
    constexpr int NROOTS = (LI_CEIL+LJ+LK)/2 + 1;
    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;

    constexpr int di = NROOTS;
    constexpr int dj = di * (LI_CEIL + 1);
    constexpr int dk = dj * (LJ + 1);
    constexpr int g_size = dk * (LK + 1);

    int nao = jk.nao;
    
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;

    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    if (rhoj == NULL){
        for (int kp = 0; kp < nfk; ++kp) {
            for (int jp = 0; jp < nfj; ++jp) {
                for (int ip = 0; ip < nfi; ++ip) {
                    
                    int loc_k = c_l_locs[LK] + kp;
                    int loc_j = c_l_locs[LJ] + jp;
                    int loc_i = c_l_locs[LI] + ip;

                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }

                    int ii = 3*ip;
                    int off_rhok = (i0+ip) + nao*(j0+jp) + (k0+kp)*nao*nao;
                    double rhok_tmp = rhok[off_rhok];
                    k3[ii + 0] += rhok_tmp * sx;
                    k3[ii + 1] += rhok_tmp * sy;
                    k3[ii + 2] += rhok_tmp * sz;
                }
            }
        }
        return;
    }

    if (rhok == NULL){
        for (int ip = 0; ip < nfi; ++ip) {
            for (int jp = 0; jp < nfj; ++jp) {
                double jx = 0.0;
                double jy = 0.0;
                double jz = 0.0;
                for (int kp = 0; kp < nfk; ++kp) {
                    int loc_j = c_l_locs[LJ] + jp;
                    int loc_i = c_l_locs[LI] + ip;
                    int loc_k = c_l_locs[LK] + kp;
                    
                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }

                    double rhoj_k = rhoj[kp + k0];
                    jx += rhoj_k * sx;
                    jy += rhoj_k * sy;
                    jz += rhoj_k * sz;
                }
                int ii = 3*ip;
                int off_dm = (ip + i0) + nao*(j0 + jp);
                //double rhoj_tmp = dm[off_dm] * rhoj_k;
                double dm_ij = dm[off_dm];
                j3[ii + 0] += jx * dm_ij;
                j3[ii + 1] += jy * dm_ij;
                j3[ii + 2] += jz * dm_ij;
            }
        }
        return;
    }

    for (int ip = 0; ip < nfi; ++ip) {
        for (int jp = 0; jp < nfj; ++jp) {
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            for (int kp = 0; kp < nfk; ++kp) {                
                int loc_k = c_l_locs[LK] + kp;
                int loc_j = c_l_locs[LJ] + jp;
                int loc_i = c_l_locs[LI] + ip;
                
                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    double gx = g[ix+ir];
                    double gy = g[iy+ir];
                    double gz = g[iz+ir];
                    sx += f[ix + ir] * gy * gz;
                    sy += gx * f[iy + ir] * gz;
                    sz += gx * gy * f[iz + ir];
                }

                int off_rhok = (i0 + ip) + nao*(j0 + jp) + (k0 + kp)*nao*nao;
                double rhok_tmp = rhok[off_rhok];
                int ii = 3*ip;
                k3[ii + 0] += rhok_tmp * sx;
                k3[ii + 1] += rhok_tmp * sy;
                k3[ii + 2] += rhok_tmp * sz;

                double rhoj_k = rhoj[k0+kp];
                jx += rhoj_k * sx;
                jy += rhoj_k * sy;
                jz += rhoj_k * sz;
            }
            int off_dm = (i0 + ip) + nao*(j0 + jp);
            //double rhoj_tmp = dm[off_dm] * rhoj_k;
            double dm_ij = dm[off_dm];
            int ii = 3*ip;
            
            j3[ii + 0] += jx * dm_ij;
            j3[ii + 1] += jy * dm_ij;
            j3[ii + 2] += jz * dm_ij;
        }
    }
}

/*
ij,  k,   nijk  -> nk
dm, rhoj, int3c_ip2 -> vjaux

ijk,     nijk   -> nk
rhok, int3c_ip2 -> vkaux
*/
template <int NROOTS> __device__
static void GINTkernel_int3c2e_ip2_getjk_direct(GINTEnvVars envs, JKMatrix jk, double* j3, double* k3, double* f, double *g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    int g_size = envs.g_size;
    int nao = jk.nao;
    int i, j, k;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;

    int i_l = envs.i_l;
    int j_l = envs.j_l;
    int k_l = envs.k_l;
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    if (rhoj == NULL){
        for (k = k0; k < k1; ++k) {
            int kp = k - k0;
            for (j = j0; j < j1; ++j) {
                int jp = j - j0;
                for (i = i0; i < i1; ++i) {
                    int ip = i - i0;

                    int loc_k = c_l_locs[k_l] + kp;
                    int loc_j = c_l_locs[j_l] + jp;
                    int loc_i = c_l_locs[i_l] + ip;

                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }

                    int kk = 3*(k-k0);
                    int off_rhok = i + nao*j + k*nao*nao;
                    double rhok_tmp = rhok[off_rhok];
                    k3[kk + 0] += sx * rhok_tmp;
                    k3[kk + 1] += sy * rhok_tmp;
                    k3[kk + 2] += sz * rhok_tmp;
                }
            }
        }
        return;
    }

    if (rhok == NULL){
        for (int kp = 0; kp < k1-k0; ++kp) {
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            
            for (int jp = 0; jp < j1-j0; ++jp) {
                for (int ip = 0; ip < i1-i0; ++ip) {
                    int loc_k = c_l_locs[k_l] + kp;
                    int loc_j = c_l_locs[j_l] + jp;
                    int loc_i = c_l_locs[i_l] + ip;

                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }
                    int off_dm = (ip+i0) + nao*(jp+j0);
                    double dm_ij = dm[off_dm];
                    jx += dm_ij * sx;
                    jy += dm_ij * sy;
                    jz += dm_ij * sz;
                }
            }
            double rhoj_k = rhoj[kp + k0];
            int kk = 3*kp;
            j3[kk + 0] += jx * rhoj_k;
            j3[kk + 1] += jy * rhoj_k;
            j3[kk + 2] += jz * rhoj_k;
        }
        return;
    }

    for (k = k0; k < k1; ++k) {
        int kp = k - k0;
        int kk = 3*(k-k0);
        double jx = 0.0;
        double jy = 0.0;
        double jz = 0.0;
        int loc_k = c_l_locs[k_l] + kp;
        for (j = j0; j < j1; ++j) {
            int jp = j - j0;
            for (i = i0; i < i1; ++i) {
                int ip = i - i0;
                int loc_j = c_l_locs[j_l] + jp;
                int loc_i = c_l_locs[i_l] + ip;

                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    double gx = g[ix+ir];
                    double gy = g[iy+ir];
                    double gz = g[iz+ir];
                    sx += f[ix + ir] * gy * gz;
                    sy += gx * f[iy + ir] * gz;
                    sz += gx * gy * f[iz + ir];
                }

                int off_rhok = i + nao*j + k*nao*nao;
                double rhok_tmp = rhok[off_rhok];
                k3[kk + 0] += sx * rhok_tmp;
                k3[kk + 1] += sy * rhok_tmp;
                k3[kk + 2] += sz * rhok_tmp;

                int off_dm = i + nao*j;
                double dm_ij = dm[off_dm];
                jx += dm_ij * sx;
                jy += dm_ij * sy;
                jz += dm_ij * sz;
            }
        }
        double rhoj_k = rhoj[k];
        j3[kk + 0] += jx * rhoj_k;
        j3[kk + 1] += jy * rhoj_k;
        j3[kk + 2] += jz * rhoj_k;
    }
}

template <int LI, int LJ, int LK> __device__
static void GINTkernel_int3c2e_ip2_getjk_direct(GINTEnvVars envs, JKMatrix jk, double* j3, double* k3, double* f, double *g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;

    constexpr int LK_CEIL = LK + 1;
    constexpr int NROOTS = (LI+LJ+LK_CEIL)/2 + 1;
    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2; 

    constexpr int di = NROOTS;
    constexpr int dj = di * (LI + 1);
    constexpr int dk = dj * (LJ + 1);
    constexpr int g_size = dk * (LK_CEIL + 1);

    int nao = jk.nao;
    
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;

    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    if (rhoj == NULL){
        for (int kp = 0; kp < nfk; ++kp) {
            for (int jp = 0; jp < nfj; ++jp) {
                for (int ip = 0; ip < nfi; ++ip) {
                    int loc_k = c_l_locs[LK] + kp;
                    int loc_j = c_l_locs[LJ] + jp;
                    int loc_i = c_l_locs[LI] + ip;

                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }

                    int kk = 3*kp;
                    int off_rhok = (i0+ip) + nao*(j0+jp) + (k0+kp)*nao*nao;
                    double rhok_tmp = rhok[off_rhok];
                    k3[kk + 0] += sx * rhok_tmp;
                    k3[kk + 1] += sy * rhok_tmp;
                    k3[kk + 2] += sz * rhok_tmp;
                }
            }
        }
        return;
    }

    if (rhok == NULL){
        for (int kp = 0; kp < nfk; ++kp) {
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            
            for (int jp = 0; jp < nfj; ++jp) {
                for (int ip = 0; ip < nfi; ++ip) {
                    int loc_k = c_l_locs[LK] + kp;
                    int loc_j = c_l_locs[LJ] + jp;
                    int loc_i = c_l_locs[LI] + ip;

                    int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                    int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                    int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                    double sx = 0.0;
                    double sy = 0.0;
                    double sz = 0.0;
#pragma unroll
                    for (int ir = 0; ir < NROOTS; ++ir){
                        double gx = g[ix+ir];
                        double gy = g[iy+ir];
                        double gz = g[iz+ir];
                        sx += f[ix + ir] * gy * gz;
                        sy += gx * f[iy + ir] * gz;
                        sz += gx * gy * f[iz + ir];
                    }
                    int off_dm = (ip+i0) + nao*(jp+j0);
                    double dm_ij = dm[off_dm];
                    jx += dm_ij * sx;
                    jy += dm_ij * sy;
                    jz += dm_ij * sz;
                }
            }
            double rhoj_k = rhoj[kp + k0];
            int kk = 3*kp;
            j3[kk + 0] += jx * rhoj_k;
            j3[kk + 1] += jy * rhoj_k;
            j3[kk + 2] += jz * rhoj_k;
        }
        return;
    }

    for (int kp = 0; kp < nfk; ++kp) {
        double jx = 0.0;
        double jy = 0.0;
        double jz = 0.0;
        
        for (int jp = 0; jp < nfj; ++jp) {
            for (int ip = 0; ip < nfi; ++ip) {
                int loc_k = c_l_locs[LK] + kp;
                int loc_j = c_l_locs[LJ] + jp;
                int loc_i = c_l_locs[LI] + ip;

                int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    double gx = g[ix+ir];
                    double gy = g[iy+ir];
                    double gz = g[iz+ir];
                    sx += f[ix + ir] * gy * gz;
                    sy += gx * f[iy + ir] * gz;
                    sz += gx * gy * f[iz + ir];
                }

                int off_rhok = (i0+ip) + nao*(j0+jp) + (k0+kp)*nao*nao;
                double rhok_tmp = rhok[off_rhok];
                k3[3*kp + 0] += sx * rhok_tmp;
                k3[3*kp + 1] += sy * rhok_tmp;
                k3[3*kp + 2] += sz * rhok_tmp;

                int off_dm = (i0+ip) + nao*(j0+jp);
                double dm_ij = dm[off_dm];
                jx += dm_ij * sx;
                jy += dm_ij * sy;
                jz += dm_ij * sz;
            }
        }
        
        double rhoj_k = rhoj[kp+k0];
        j3[3*kp + 0] += jx * rhoj_k;
        j3[3*kp + 1] += jy * rhoj_k;
        j3[3*kp + 2] += jz * rhoj_k;
    }
}

__device__
static void write_int3c2e_ip1_jk(JKMatrix jk, double* j3, double* k3, int ish){
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    double *vj = jk.vj;
    double *vk = jk.vk;
    int nao = jk.nao;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];

    if (vj != NULL){
        for (int i = i0; i < i1; ++i){
            for (int j = 0; j < 3; j++){
                int ii = 3*(i-i0) + j;
                sdata[tx][ty] = j3[ii]; __syncthreads();
                if(THREADSY >= 16 && ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
                if(THREADSY >= 8  && ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
                if(THREADSY >= 4  && ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
                if(THREADSY >= 2  && ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
                if (ty == 0) atomicAdd(vj+i+j*nao, sdata[tx][0]);
            }
        }
    }

    if (vk != NULL){
        for (int i = i0; i < i1; ++i){
            for (int j = 0; j < 3; j++){
                int ii = 3*(i-i0) + j;
                sdata[tx][ty] = k3[ii]; __syncthreads();
                if(THREADSY >= 16 && ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
                if(THREADSY >= 8  && ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
                if(THREADSY >= 4  && ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
                if(THREADSY >= 2  && ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
                if (ty == 0) atomicAdd(vk+i+j*nao, sdata[tx][0]);
            }
        }
    }
}

__device__
static void write_int3c2e_ip2_jk(JKMatrix jk, double *j3, double* k3, int ksh){
    int *ao_loc = c_bpcache.ao_loc;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    double *vj = jk.vj;
    double *vk = jk.vk;
    int naux = jk.naux;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];

    if (vj != NULL){
        for (int k = k0; k < k1; ++k){
            for (int j = 0; j < 3; j++){
                int kk = 3*(k-k0) + j;
                sdata[tx][ty] = j3[kk]; __syncthreads();
                if(THREADSX >= 16 && tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
                if(THREADSX >= 8  && tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
                if(THREADSX >= 4  && tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
                if(THREADSX >= 2  && tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
                if (tx == 0) atomicAdd(vj+k+j*naux, sdata[0][ty]);
            }
        }
    }
    if (vk != NULL){
        for (int k = k0; k < k1; ++k){
            for (int j = 0; j < 3; j++){
                int kk = 3*(k-k0) + j;
                sdata[tx][ty] = k3[kk]; __syncthreads();
                if(THREADSX >= 16 && tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
                if(THREADSX >= 8  && tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
                if(THREADSX >= 4  && tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
                if(THREADSX >= 2  && tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
                if (tx == 0) atomicAdd(vk+k+j*naux, sdata[0][ty]);
            }
        }
    }
}


