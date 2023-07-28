/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "gint/g2e.h"
#include "gint/cint2e.cuh"
#include "gint/gout2e.cuh"
#include "gint/g2e.cu"

// this is not supposed to be exectued
template <int NROOTS, int GOUTSIZE> __global__
static void GINTint2e_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];

    int task_id = task_ij + ntasks_ij * task_kl;
    double *uw = envs.uw + task_id * nprim_ij * nprim_kl * NROOTS * 2;
    double gout[GOUTSIZE];
    double *g = gout + envs.nf;
    int i;
    for (i = 0; i < envs.nf; ++i) {
        gout[i] = 0;
    }

    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_2e_2d4d<NROOTS>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTgout2e<NROOTS>(envs, gout, g);
        uw += NROOTS * 2;
    } }
    //GINTkernel_getjk(jk, gout, ish, jsh, ksh, lsh);
}

__global__
static void GINTint2e_jk_kernel0000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int *ao_loc = c_bpcache.ao_loc;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    double gout0 = 0;
    
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);
        if (x > 3.e-7) {
            double tt = sqrt(x);
            double fmt0 = SQRTPIE4 / tt * erf(tt);
            fac *= fmt0;
        }
        gout0 += fac;
    } }

    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            atomicAdd(vj+k0+nao*l0, gout0*dm[i0+nao*j0]);
            atomicAdd(vj+i0+nao*j0, gout0*dm[k0+nao*l0]);
            vj += nao2;
        }
        if (vk != NULL) {
            atomicAdd(vk+i0+nao*k0, gout0*dm[j0+nao*l0]);
            atomicAdd(vk+i0+nao*l0, gout0*dm[j0+nao*k0]);
            atomicAdd(vk+j0+nao*k0, gout0*dm[i0+nao*l0]);
            atomicAdd(vk+j0+nao*l0, gout0*dm[i0+nao*k0]);
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_jk_kernel1000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int *ao_loc = c_bpcache.ao_loc;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;

    int ij, kl, i_dm;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
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
        //double fac = eij * ekl / (sqrt(aijkl) * a1);
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
        gout0 += g_1 * g_2 * g_4;
        gout1 += g_0 * g_3 * g_4;
        gout2 += g_0 * g_2 * g_5;
    } }

    double d_0, d_1, d_2;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            vk += nao2;
        }
        dm += nao2;
    }
}

#if POLYFIT_ORDER >= 3
template <> __global__
void GINTint2e_jk_kernel<3, GSIZE3>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double log_q_ij = offsets.log_q_ij[task_ij];
    double log_q_kl = offsets.log_q_kl[task_kl];
    if (is_skip(jk, log_q_ij, log_q_kl, ish, jsh, ksh, lsh, offsets.log_cutoff)){
        return;
    }

    double uw[6];
    double g[GSIZE3];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
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
        GINTrys_root3(x, uw);
        GINTscale_u<3>(uw, theta);
        GINTg0_2e_2d4d<3>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTkernel_direct_getjk<3, GSIZE3>(envs, jk, g, ish, jsh, ksh, lsh);
    } }
}
#endif

#if POLYFIT_ORDER >= 4
template <> __global__
void GINTint2e_jk_kernel<4, GSIZE4>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double log_q_ij = offsets.log_q_ij[task_ij];
    double log_q_kl = offsets.log_q_kl[task_kl];
    if (is_skip(jk, log_q_ij, log_q_kl, ish, jsh, ksh, lsh, offsets.log_cutoff)){
        return;
    }

    double uw[8];
    double g[GSIZE4];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
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
        GINTrys_root4(x, uw);
        GINTscale_u<4>(uw, theta);
        GINTg0_2e_2d4d<4>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTkernel_direct_getjk<4, GSIZE4>(envs, jk, g, ish, jsh, ksh, lsh);
    } }
}
#endif

#if POLYFIT_ORDER >= 5
template <> __global__
void GINTint2e_jk_kernel<5, GSIZE5>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double log_q_ij = offsets.log_q_ij[task_ij];
    double log_q_kl = offsets.log_q_kl[task_kl];
    if (is_skip(jk, log_q_ij, log_q_kl, ish, jsh, ksh, lsh, offsets.log_cutoff)){
        return;
    }
    double uw[10];
    double g[GSIZE5];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
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
        GINTrys_root5(x, uw);
        GINTscale_u<5>(uw, theta);
        GINTg0_2e_2d4d<5>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTkernel_direct_getjk<5, GSIZE5>(envs, jk, g, ish, jsh, ksh, lsh);
    } }
}
#endif


#if POLYFIT_ORDER >= 6
template <> __global__
void GINTint2e_jk_kernel<6, GSIZE6>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double log_q_ij = offsets.log_q_ij[task_ij];
    double log_q_kl = offsets.log_q_kl[task_kl];
    if (is_skip(jk, log_q_ij, log_q_kl, ish, jsh, ksh, lsh, offsets.log_cutoff)){
        return;
    }
    double uw[12];
    double g[GSIZE6];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
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
        GINTrys_root6(x, uw);
        GINTscale_u<6>(uw, theta);
        GINTg0_2e_2d4d<6>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTkernel_direct_getjk<6, GSIZE6>(envs, jk, g, ish, jsh, ksh, lsh);
    } }
}
#endif


#if POLYFIT_ORDER >= 7
template <> __global__
void GINTint2e_jk_kernel<7, GSIZE7>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double log_q_ij = offsets.log_q_ij[task_ij];
    double log_q_kl = offsets.log_q_kl[task_kl];
    if (is_skip(jk, log_q_ij, log_q_kl, ish, jsh, ksh, lsh, offsets.log_cutoff)){
        return;
    }
    double uw[14];
    double g[GSIZE7];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
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
        GINTrys_root7(x, uw);
        GINTscale_u<7>(uw, theta);
        GINTg0_2e_2d4d<7>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTkernel_direct_getjk<7, GSIZE7>(envs, jk, g, ish, jsh, ksh, lsh);
    } }
}
#endif


#if POLYFIT_ORDER >= 8
template <> __global__
void GINTint2e_jk_kernel<8, GSIZE8>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double log_q_ij = offsets.log_q_ij[task_ij];
    double log_q_kl = offsets.log_q_kl[task_kl];
    if (is_skip(jk, log_q_ij, log_q_kl, ish, jsh, ksh, lsh, offsets.log_cutoff)){
        return;
    }
    double uw[16];
    double g[GSIZE8];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
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
        GINTrys_root8(x, uw);
        GINTscale_u<8>(uw, theta);
        GINTg0_2e_2d4d<8>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTkernel_direct_getjk<8, GSIZE8>(envs, jk, g, ish, jsh, ksh, lsh);
    } }
}
#endif


#if POLYFIT_ORDER >= 9
template <> __global__
void GINTint2e_jk_kernel<9, GSIZE9>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double log_q_ij = offsets.log_q_ij[task_ij];
    double log_q_kl = offsets.log_q_kl[task_kl];
    if (is_skip(jk, log_q_ij, log_q_kl, ish, jsh, ksh, lsh, offsets.log_cutoff)){
        return;
    }
    double uw[18];
    double g[GSIZE9];

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
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
        GINTrys_root9(x, uw);
        GINTscale_u<9>(uw, theta);
        GINTg0_2e_2d4d<9>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTkernel_direct_getjk<9, GSIZE9>(envs, jk, g, ish, jsh, ksh, lsh);
    } }
}
#endif
