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
#include "gint/cint2e.hpp"
#include "gvhf.h"
/*
__attribute__((always_inline))
static void GINTkernel_int3c2e_ip1_getjk(JKMatrix jk, double* __restrict__ gout,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.get().ao_loc;
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

__attribute__((always_inline))
static void GINTkernel_int3c2e_ip2_getjk(JKMatrix jk, double* __restrict__ gout,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.get().ao_loc;
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
template <int NROOTS> __attribute__((always_inline))
static void GINTkernel_int3c2e_getj_pass1(GINTEnvVars envs, JKMatrix jk, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.get().ao_loc;
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
template <int NROOTS> __attribute__((always_inline))
static void GINTkernel_int3c2e_getj_pass2(GINTEnvVars envs, JKMatrix jk, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.get().ao_loc;
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

template <int NROOTS> __attribute__((always_inline))
static void GINTkernel_int3c2e_ip1_getjk_direct(GINTEnvVars envs, JKMatrix jk, double* j3, double* k3, double* f, double* g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.get().ao_loc;
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
            int ip = i - i0;
            for (j = j0; j < j1; ++j) {
                int jp = j - j0;
                double jx = 0.0;
                double jy = 0.0;
                double jz = 0.0;
                int loc_j = c_l_locs[j_l] + jp;
                int loc_i = c_l_locs[i_l] + ip;
                for (k = k0; k < k1; ++k) {
                    int kp = k - k0;
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
        int ip = i - i0;
        int ii = 3*(i-i0);
        for (j = j0; j < j1; ++j) {
            int jp = j - j0;
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            int loc_j = c_l_locs[j_l] + jp;
            int loc_i = c_l_locs[i_l] + ip;
            for (k = k0; k < k1; ++k) {
                int kp = k - k0;
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

                int off_rhok = i + nao*j + k*nao*nao;
                double rhok_tmp = rhok[off_rhok];
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
template <int NROOTS> __attribute__((always_inline))
static void GINTkernel_int3c2e_ip2_getjk_direct(GINTEnvVars envs, JKMatrix jk, double* j3, double* k3, double* f, double *g,
                       int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.get().ao_loc;
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
            //int kp = k - k0;
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            int loc_k = c_l_locs[k_l] + kp;
            for (int jp = 0; jp < j1-j0; ++jp) {
                //int jp = j - j0;
                for (int ip = 0; ip < i1-i0; ++ip) {
                    //int ip = i - i0;

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

__attribute__((always_inline))
static void write_int3c2e_ip1_jk(JKMatrix jk, double* j3, double* k3, int ish, sycl::nd_item<2>& item){
    int *ao_loc = c_bpcache.get().ao_loc;
    int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    double *vj = jk.vj;
    double *vk = jk.vk;
    int nao = jk.nao;

    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    using tile_t = double[THREADSX][THREADSY];
    tile_t& sdata = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(item.get_group());

    if (vj != NULL){
        for (int i = i0; i < i1; ++i){
            for (int j = 0; j < 3; j++){
                int ii = 3*(i-i0) + j;
                sdata[tx][ty] = j3[ii]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<8) sdata[tx][ty] += sdata[tx][ty+8]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<4) sdata[tx][ty] += sdata[tx][ty+4]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<2) sdata[tx][ty] += sdata[tx][ty+2]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<1) sdata[tx][ty] += sdata[tx][ty+1]; item.barrier(sycl::access::fence_space::local_space);
                if (ty == 0) atomicAdd(vj+i+j*nao, sdata[tx][0]);
            }
        }
    }

    if (vk != NULL){
        for (int i = i0; i < i1; ++i){
            for (int j = 0; j < 3; j++){
                int ii = 3*(i-i0) + j;
                sdata[tx][ty] = k3[ii]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<8) sdata[tx][ty] += sdata[tx][ty+8]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<4) sdata[tx][ty] += sdata[tx][ty+4]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<2) sdata[tx][ty] += sdata[tx][ty+2]; item.barrier(sycl::access::fence_space::local_space);
                if(ty<1) sdata[tx][ty] += sdata[tx][ty+1]; item.barrier(sycl::access::fence_space::local_space);
                if (ty == 0) atomicAdd(vk+i+j*nao, sdata[tx][0]);
            }
        }
    }
}

__attribute__((always_inline))
static void write_int3c2e_ip2_jk(JKMatrix jk, double *j3, double* k3, int ksh, sycl::nd_item<2>& item){
    int *ao_loc = c_bpcache.get().ao_loc;
    int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    double *vj = jk.vj;
    double *vk = jk.vk;
    int naux = jk.naux;

    int tx = item.get_local_id(1);
    int ty = item.get_local_id(0);
    using tile_t = double[THREADSX][THREADSY];
    tile_t& sdata = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(item.get_group());

    if (vj != NULL){
        for (int k = k0; k < k1; ++k){
            for (int j = 0; j < 3; j++){
                int kk = 3*(k-k0) + j;
                sdata[tx][ty] = j3[kk]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; item.barrier(sycl::access::fence_space::local_space);
                if (tx == 0) atomicAdd(vj+k+j*naux, sdata[0][ty]);
            }
        }
    }
    if (vk != NULL){
        for (int k = k0; k < k1; ++k){
            for (int j = 0; j < 3; j++){
                int kk = 3*(k-k0) + j;
                sdata[tx][ty] = k3[kk]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; item.barrier(sycl::access::fence_space::local_space);
                if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; item.barrier(sycl::access::fence_space::local_space);
                if (tx == 0) atomicAdd(vk+k+j*naux, sdata[0][ty]);
            }
        }
    }
}
