/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
    int i, j, k, off_dm, off_rhok;
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
                
                    int ii = 3*(i-i0);
                    off_rhok = i + nao*j + k*nao*nao;
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
        for (k = k0; k < k1; ++k) {
            int kp = k - k0;
            double rhoj_k = rhoj[k];
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
                    int ii = 3*(i-i0);
                    off_dm = i + nao*j;
                    double rhoj_tmp = dm[off_dm] * rhoj_k;
                    j3[ii + 0] += rhoj_tmp * sx;
                    j3[ii + 1] += rhoj_tmp * sy;
                    j3[ii + 2] += rhoj_tmp * sz;
                }
            }
        }
        return;
    }

    for (k = k0; k < k1; ++k) {
        int kp = k - k0;
        double rhoj_k = rhoj[k];
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
                int ii = 3*(i-i0);
                off_dm = i + nao*j;
                double rhoj_tmp = dm[off_dm] * rhoj_k;
                j3[ii + 0] += rhoj_tmp * sx;
                j3[ii + 1] += rhoj_tmp * sy;
                j3[ii + 2] += rhoj_tmp * sz;

                off_rhok = i + nao*j + k*nao*nao;
                double rhok_tmp = rhok[off_rhok];
                k3[ii + 0] += rhok_tmp * sx;
                k3[ii + 1] += rhok_tmp * sy;
                k3[ii + 2] += rhok_tmp * sz;
            }
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
    int i, j, k, off_dm, off_rhok;
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
                    off_rhok = i + nao*j + k*nao*nao;
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
        for (k = k0; k < k1; ++k) {
            int kp = k - k0;
            double rhoj_k = rhoj[k];
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
                    off_dm = i + nao*j;
                    double rhoj_tmp = dm[off_dm] * rhoj_k;
                    j3[kk + 0] += sx * rhoj_tmp;
                    j3[kk + 1] += sy * rhoj_tmp;
                    j3[kk + 2] += sz * rhoj_tmp;
                }
            }
        }
        return;
    }

    for (k = k0; k < k1; ++k) {
        int kp = k - k0;
        double rhoj_k = rhoj[k];
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
                off_dm = i + nao*j;
                double rhoj_tmp = dm[off_dm] * rhoj_k;
                j3[kk + 0] += sx * rhoj_tmp;
                j3[kk + 1] += sy * rhoj_tmp;
                j3[kk + 2] += sz * rhoj_tmp;
                
                off_rhok = i + nao*j + k*nao*nao;
                double rhok_tmp = rhok[off_rhok];
                k3[kk + 0] += sx * rhok_tmp;
                k3[kk + 1] += sy * rhok_tmp;
                k3[kk + 2] += sz * rhok_tmp;
            }
        }
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
                if(ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
                if(ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
                if(ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
                if(ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
                if (ty == 0) atomicAdd(vj+i+j*nao, sdata[tx][0]);
            }
        }
    }

    if (vk != NULL){
        for (int i = i0; i < i1; ++i){
            for (int j = 0; j < 3; j++){
                int ii = 3*(i-i0) + j;
                sdata[tx][ty] = k3[ii]; __syncthreads();
                if(ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
                if(ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
                if(ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
                if(ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
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
                if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
                if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
                if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
                if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
                if (tx == 0) atomicAdd(vj+k+j*naux, sdata[0][ty]);
            }
        }
    }
    if (vk != NULL){
        for (int k = k0; k < k1; ++k){
            for (int j = 0; j < 3; j++){
                int kk = 3*(k-k0) + j;
                sdata[tx][ty] = k3[kk]; __syncthreads();
                if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
                if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
                if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
                if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
                if (tx == 0) atomicAdd(vk+k+j*naux, sdata[0][ty]);
            }
        }
    }
}

__global__
static void GINTrun_int3c2e_ip1_jk_kernel1000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double* __restrict__ a1 = c_bpcache.a1;
    int ij, kl;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
        double ai2 = -2.0*a1[ij];
        double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
        double ekl = e12[kl];
        double xkl = x12[kl];
        double ykl = y12[kl];
        double zkl = z12[kl];
        double xijxkl = xij - xkl;
        double yijykl = yij - ykl;
        double zijzkl = zij - zkl;
        double aijkl = aij + akl;
        double a1 = aij * akl;
        double a0 = a1 / aijkl;
        double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
        a0 *= theta;
        double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
        double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
        double root0, weight0;
        if (x < 3.e-7) {
            root0 = 0.5;
            weight0 = 1.;
        } else {
            double tt = sqrt(x);
            double fmt0 = SQRTPIE4 / tt * erf(tt);
            weight0 = fmt0;
            double e = exp(-x);
            double b = .5 / x;
            double fmt1 = b * (fmt0 - e);
            root0 = fmt1 / (fmt0 - fmt1);
        }
        root0 /= root0 + 1 - root0 * theta;
        double u2 = a0 * root0;
        double tmp2 = akl * u2 / (u2 * aijkl + a1);;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = 1;
        double g_3 = c00y;
        double g_4 = norm * fac * weight0;
        double g_5 = g_4 * c00z;
        
        double f_1 = ai2 * g_1;
        double f_3 = ai2 * g_3;
        double f_5 = ai2 * g_5;
        
        gout0 += f_1 * g_2 * g_4;
        gout1 += g_0 * f_3 * g_4;
        gout2 += g_0 * g_2 * f_5;
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh] - jk.ao_offsets_k;
    
    int nao = jk.nao;
    double* __restrict__ dm = jk.dm;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ vj = jk.vj;
    double* __restrict__ vk = jk.vk;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];
    if (!active){
        gout0 = 0.0; gout1 = 0.0; gout2 = 0.0;
    }
    if (vj != NULL){
        double rhoj_tmp;
        int off_dm = i0 + nao*j0;
        rhoj_tmp = dm[off_dm] * rhoj[k0];
        double vj_tmp[3];
        vj_tmp[0] = gout0*rhoj_tmp;
        vj_tmp[1] = gout1*rhoj_tmp;
        vj_tmp[2] = gout2*rhoj_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vj_tmp[j]; __syncthreads();
            if(ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
            if(ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
            if(ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
            if(ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
            if (ty == 0) atomicAdd(vj+i0+j*nao, sdata[tx][0]);
        }
    }
    if (vk != NULL){
        double rhok_tmp;
        int off_rhok = i0 + nao*j0 + k0*nao*nao;
        rhok_tmp = rhok[off_rhok];
        double vk_tmp[3];
        vk_tmp[0] = gout0 * rhok_tmp;
        vk_tmp[1] = gout1 * rhok_tmp;
        vk_tmp[2] = gout2 * rhok_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vk_tmp[j]; __syncthreads();
            if(ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
            if(ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
            if(ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
            if(ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
            if (ty == 0) atomicAdd(vk+i0+j*nao, sdata[tx][0]);
        }
    }
}


__global__
static void GINTrun_int3c2e_ip2_jk_kernel0010(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double* __restrict__ a1 = c_bpcache.a1;
    int ij, kl;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
        double ak2 = -2.0*a1[kl];
        double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
        double ekl = e12[kl];
        double xkl = x12[kl];
        double ykl = y12[kl];
        double zkl = z12[kl];
        double xijxkl = xij - xkl;
        double yijykl = yij - ykl;
        double zijzkl = zij - zkl;
        double aijkl = aij + akl;
        double a1 = aij * akl;
        double a0 = a1 / aijkl;
        double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
        a0 *= theta;
        double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
        double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
        double root0, weight0;
        if (x < 3.e-7) {
            root0 = 0.5;
            weight0 = 1.;
        } else {
            double tt = sqrt(x);
            double fmt0 = SQRTPIE4 / tt * erf(tt);
            weight0 = fmt0;
            double e = exp(-x);
            double b = .5 / x;
            double fmt1 = b * (fmt0 - e);
            root0 = fmt1 / (fmt0 - fmt1);
        }
        root0 /= root0 + 1 - root0 * theta;
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c0px;
        double g_2 = 1;
        double g_3 = c0py;
        double g_4 = weight0 * fac;
        double g_5 = c0pz * g_4;

        double f_1 = ak2 * g_1;
        double f_3 = ak2 * g_3;
        double f_5 = ak2 * g_5;

        gout0 += f_1 * g_2 * g_4;
        gout1 += g_0 * f_3 * g_4;
        gout2 += g_0 * g_2 * f_5;

    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh] - jk.ao_offsets_k;

    int nao = jk.nao;
    int naux = jk.naux;
    double* __restrict__ dm = jk.dm;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ vj = jk.vj;
    double* __restrict__ vk = jk.vk;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];
    if (!active){
        gout0 = 0.0; gout1 = 0.0; gout2 = 0.0;
    }
    if (vj != NULL){
        double rhoj_tmp;
        int off_dm = i0 + nao*j0;
        rhoj_tmp = dm[off_dm] * rhoj[k0];
        double vj_tmp[3];
        vj_tmp[0] = gout0 * rhoj_tmp;
        vj_tmp[1] = gout1 * rhoj_tmp;
        vj_tmp[2] = gout2 * rhoj_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vj_tmp[j]; __syncthreads();
            if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
            if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
            if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
            if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
            if (tx == 0) atomicAdd(vj+k0+j*naux, sdata[0][ty]);
        }
    }

    if (vk != NULL){
        double rhok_tmp;
        int off_rhok = i0 + nao*j0 + k0*nao*nao;
        rhok_tmp = rhok[off_rhok];
        double vk_tmp[3];
        vk_tmp[0] = gout0 * rhok_tmp;
        vk_tmp[1] = gout1 * rhok_tmp;
        vk_tmp[2] = gout2 * rhok_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vk_tmp[j]; __syncthreads();
            if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
            if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
            if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
            if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
            if (tx == 0) atomicAdd(vk+k0+j*naux, sdata[0][ty]);
        }
    }
}
