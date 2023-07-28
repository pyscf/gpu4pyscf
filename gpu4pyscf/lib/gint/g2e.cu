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
#include "g2e.h"
#include "cint2e.cuh"

// TODO: prime basis into different thread

template <int NROOTS> __device__
static void GINTg0_2e_2d4d(GINTEnvVars envs, double* __restrict__ g, double*__restrict__ uw, double norm,
                    int ish, int jsh, int ksh, int lsh, int prim_ij, int prim_kl)
{
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double aij = a12[prim_ij];
    double akl = a12[prim_kl];
    double eij = e12[prim_ij];
    double ekl = e12[prim_kl];
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double omega = envs.omega;
    double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
    a0 *= theta;
    //double fac = eij * ekl / (sqrt(aijkl) * a1);
    double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
    
    double* __restrict__ u = uw;
    double* __restrict__ w = u + NROOTS;
    double* __restrict__ gx = g;
    double* __restrict__ gy = g + envs.g_size;
    double* __restrict__ gz = g + envs.g_size * 2;

    double xij = x12[prim_ij];
    double yij = y12[prim_ij];
    double zij = z12[prim_ij];
    double xkl = x12[prim_kl];
    double ykl = y12[prim_kl];
    double zkl = z12[prim_kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;
    double xixj, yiyj, zizj, xkxl, ykyl, zkzl;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xijxi = xij - xi;
    double yijyi = yij - yi;
    double zijzi = zij - zi;
    double xklxk = xkl - xk;
    double yklyk = ykl - yk;
    double zklzk = zkl - zk;

    int nmax = envs.li_ceil + envs.lj_ceil;
    int mmax = envs.lk_ceil + envs.ll_ceil;
    int ijmin = envs.ijmin;
    int klmin = envs.klmin;
    int dm = envs.stride_klmax;
    int dn = envs.stride_ijmax;
    int di = envs.stride_ijmax;
    int dj = envs.stride_ijmin;
    int dk = envs.stride_klmax;
    int dl = envs.stride_klmin;
    int dij = envs.g_size_ij;
    int i, k;
    int j, l, m, n, off;
    double tmpb0;
    double s0x, s1x, s2x, t0x, t1x;
    double s0y, s1y, s2y, t0y, t1y;
    double s0z, s1z, s2z, t0z, t1z;
    double u2, tmp1, tmp2, tmp3, tmp4;
    double b00, b10, b01, c00x, c00y, c00z, c0px, c0py, c0pz;
    
    for (i = 0; i < NROOTS; ++i) {
        gx[i] = norm;
        gy[i] = fac;
        gz[i] = w[i];

        u2 = a0 * u[i];
        tmp4 = .5 / (u2 * aijkl + a1);
        b00 = u2 * tmp4;
        tmp1 = 2 * b00;
        tmp2 = tmp1 * akl;
        b10 = b00 + tmp4 * akl;
        c00x = xijxi - tmp2 * xijxkl;
        c00y = yijyi - tmp2 * yijykl;
        c00z = zijzi - tmp2 * zijzkl;
        
        if (nmax > 0) {
            // gx(irys,0,1) = c00(irys) * gx(irys,0,0)
            // gx(irys,0,n+1) = c00(irys)*gx(irys,0,n) + n*b10(irys)*gx(irys,0,n-1)
            //for (n = 1; n < nmax; ++n) {
            //    off = n * dn;
            //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
            //        gx[j+dn] = c00x[i] * gx[j] + n * b10[i] * gx[j-dn];
            //        gy[j+dn] = c00y[i] * gy[j] + n * b10[i] * gy[j-dn];
            //        gz[j+dn] = c00z[i] * gz[j] + n * b10[i] * gz[j-dn];
            //    }
            //}
            s0x = gx[i];
            s0y = gy[i];
            s0z = gz[i];
            s1x = c00x * s0x;
            s1y = c00y * s0y;
            s1z = c00z * s0z;
            gx[i+dn] = s1x;
            gy[i+dn] = s1y;
            gz[i+dn] = s1z;
            for (n = 1; n < nmax; ++n) {
                s2x = c00x * s1x + n * b10 * s0x;
                s2y = c00y * s1y + n * b10 * s0y;
                s2z = c00z * s1z + n * b10 * s0z;
                gx[i+(n+1)*dn] = s2x;
                gy[i+(n+1)*dn] = s2y;
                gz[i+(n+1)*dn] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }
        }

        if (mmax > 0) {
            // gx(irys,1,0) = c0p(irys) * gx(irys,0,0)
            // gx(irys,m+1,0) = c0p(irys)*gx(irys,m,0) + m*b01(irys)*gx(irys,m-1,0)
            //for (m = 1; m < mmax; ++m) {
            //    off = m * dm;
            //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
            //        gx[j+dm] = c0px[i] * gx[j] + m * b01[i] * gx[j-dm];
            //        gy[j+dm] = c0py[i] * gy[j] + m * b01[i] * gy[j-dm];
            //        gz[j+dm] = c0pz[i] * gz[j] + m * b01[i] * gz[j-dm];
            //    }
            //}
            tmp3 = tmp1 * aij;
            b01 = b00 + tmp4 * aij;
            c0px = xklxk + tmp3 * xijxkl;
            c0py = yklyk + tmp3 * yijykl;
            c0pz = zklzk + tmp3 * zijzkl;
            s0x = gx[i];
            s0y = gy[i];
            s0z = gz[i];
            s1x = c0px * s0x;
            s1y = c0py * s0y;
            s1z = c0pz * s0z;
            gx[i+dm] = s1x;
            gy[i+dm] = s1y;
            gz[i+dm] = s1z;
            for (m = 1; m < mmax; ++m) {
                s2x = c0px * s1x + m * b01 * s0x;
                s2y = c0py * s1y + m * b01 * s0y;
                s2z = c0pz * s1z + m * b01 * s0z;
                gx[i+(m+1)*dm] = s2x;
                gy[i+(m+1)*dm] = s2y;
                gz[i+(m+1)*dm] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }

            if (nmax > 0) {
                // gx(irys,1,1) = c0p(irys)*gx(irys,0,1) + b00(irys)*gx(irys,0,0)
                // gx(irys,m+1,1) = c0p(irys)*gx(irys,m,1)
                // + m*b01(irys)*gx(irys,m-1,1)
                // + b00(irys)*gx(irys,m,0)
                //for (m = 1; m < mmax; ++m) {
                //    off = m * dm + dn;
                //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
                //        gx[j+dm] = c0px[i]*gx[j] + m*b01[i]*gx[j-dm] + b00[i]*gx[j-dn];
                //        gy[j+dm] = c0py[i]*gy[j] + m*b01[i]*gy[j-dm] + b00[i]*gy[j-dn];
                //        gz[j+dm] = c0pz[i]*gz[j] + m*b01[i]*gz[j-dm] + b00[i]*gz[j-dn];
                //    }
                //}
                s0x = gx[i+dn];
                s0y = gy[i+dn];
                s0z = gz[i+dn];
                s1x = c0px * s0x + b00 * gx[i];
                s1y = c0py * s0y + b00 * gy[i];
                s1z = c0pz * s0z + b00 * gz[i];
                gx[i+dn+dm] = s1x;
                gy[i+dn+dm] = s1y;
                gz[i+dn+dm] = s1z;
                for (m = 1; m < mmax; ++m) {
                    s2x = c0px*s1x + m*b01*s0x + b00*gx[i+m*dm];
                    s2y = c0py*s1y + m*b01*s0y + b00*gy[i+m*dm];
                    s2z = c0pz*s1z + m*b01*s0z + b00*gz[i+m*dm];
                    gx[i+dn+(m+1)*dm] = s2x;
                    gy[i+dn+(m+1)*dm] = s2y;
                    gz[i+dn+(m+1)*dm] = s2z;
                    s0x = s1x;
                    s0y = s1y;
                    s0z = s1z;
                    s1x = s2x;
                    s1y = s2y;
                    s1z = s2z;
                }
            }
        }

        // gx(irys,m,n+1) = c00(irys)*gx(irys,m,n)
        // + n*b10(irys)*gx(irys,m,n-1)
        // + m*b00(irys)*gx(irys,m-1,n)
        for (m = 1; m <= mmax; ++m) {
            //for (n = 1; n < nmax; ++n) {
            //    off = m * dm + n * dn;
            //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
            //        gx[j+dn] = c00x[i]*gx[j] +n*b10[i]*gx[j-dn] + m*b00[i]*gx[j-dm];
            //        gy[j+dn] = c00y[i]*gy[j] +n*b10[i]*gy[j-dn] + m*b00[i]*gy[j-dm];
            //        gz[j+dn] = c00z[i]*gz[j] +n*b10[i]*gz[j-dn] + m*b00[i]*gz[j-dm];
            //    }
            //}
            off = m * dm;
            j = off + i;
            s0x = gx[j];
            s0y = gy[j];
            s0z = gz[j];
            s1x = gx[j + dn];
            s1y = gy[j + dn];
            s1z = gz[j + dn];
            tmpb0 = m * b00;
            for (n = 1; n < nmax; ++n) {
                s2x = c00x*s1x + n*b10*s0x + tmpb0*gx[j+n*dn-dm];
                s2y = c00y*s1y + n*b10*s0y + tmpb0*gy[j+n*dn-dm];
                s2z = c00z*s1z + n*b10*s0z + tmpb0*gz[j+n*dn-dm];
                gx[j+(n+1)*dn] = s2x;
                gy[j+(n+1)*dn] = s2y;
                gz[j+(n+1)*dn] = s2z;
                s0x = s1x;
                s0y = s1y;
                s0z = s1z;
                s1x = s2x;
                s1y = s2y;
                s1z = s2z;
            }
        }
    }

    if (ijmin > 0) {
        // g(i,j) = rirj * g(i,j-1) +  g(i+1,j-1)
        xixj = xi - bas_x[jsh];
        yiyj = yi - bas_y[jsh];
        zizj = zi - bas_z[jsh];
        //for (k = 0; k <= mmax; ++k) {
        //for (j = 0; j < ijmin; ++j) {
        //for (i = nmax-1-j; i >= 0; i--) {
        //    off = k*dk + j*dj + i*di;
        //    for (n = off; n < off+NROOTS; ++n) {
        //        gx[dj+n] = xixj * gx[n] + gx[di+n];
        //        gy[dj+n] = yiyj * gy[n] + gy[di+n];
        //        gz[dj+n] = zizj * gz[n] + gz[di+n];
        //    }
        //} } }

        // unrolling j
        for (j = 0; j < ijmin-1; j+=2, nmax-=2) {
        for (k = 0; k <= mmax; ++k) {
            off = k * dk + j * dj;
            for (n = off; n < off+NROOTS; ++n) {
                s0x = gx[n+nmax*di-di];
                s0y = gy[n+nmax*di-di];
                s0z = gz[n+nmax*di-di];
                t1x = xixj * s0x + gx[n+nmax*di];
                t1y = yiyj * s0y + gy[n+nmax*di];
                t1z = zizj * s0z + gz[n+nmax*di];
                gx[dj+n+nmax*di-di] = t1x;
                gy[dj+n+nmax*di-di] = t1y;
                gz[dj+n+nmax*di-di] = t1z;
                s1x = s0x;
                s1y = s0y;
                s1z = s0z;
                for (i = nmax-2; i >= 0; i--) {
                    s0x = gx[n+i*di];
                    s0y = gy[n+i*di];
                    s0z = gz[n+i*di];
                    t0x = xixj * s0x + s1x;
                    t0y = yiyj * s0y + s1y;
                    t0z = zizj * s0z + s1z;
                    gx[dj+n+i*di] = t0x;
                    gy[dj+n+i*di] = t0y;
                    gz[dj+n+i*di] = t0z;
                    gx[dj+dj+n+i*di] = xixj * t0x + t1x;
                    gy[dj+dj+n+i*di] = yiyj * t0y + t1y;
                    gz[dj+dj+n+i*di] = zizj * t0z + t1z;
                    s1x = s0x;
                    s1y = s0y;
                    s1z = s0z;
                    t1x = t0x;
                    t1y = t0y;
                    t1z = t0z;
                }
            }
        } }

        if (j < ijmin) {
            for (k = 0; k <= mmax; ++k) {
                off = k * dk + j * dj;
                for (n = off; n < off+NROOTS; ++n) {
                    s1x = gx[n + nmax*di];
                    s1y = gy[n + nmax*di];
                    s1z = gz[n + nmax*di];
                    for (i = nmax-1; i >= 0; i--) {
                        s0x = gx[n+i*di];
                        s0y = gy[n+i*di];
                        s0z = gz[n+i*di];
                        gx[dj+n+i*di] = xixj * s0x + s1x;
                        gy[dj+n+i*di] = yiyj * s0y + s1y;
                        gz[dj+n+i*di] = zizj * s0z + s1z;
                        s1x = s0x;
                        s1y = s0y;
                        s1z = s0z;
                    }
                }
            }
        }
    }

    if (klmin > 0) {
        // g(...,k,l) = rkrl * g(...,k,l-1) + g(...,k+1,l-1)
        xkxl = xk - bas_x[lsh];
        ykyl = yk - bas_y[lsh];
        zkzl = zk - bas_z[lsh];
        //for (l = 0; l < klmin; ++l) {
        //for (k = mmax-1-l; k >= 0; k--) {
        //    off = l*dl + k*dk;
        //    for (n = off; n < off+dij; ++n) {
        //        gx[dl+n] = xkxl * gx[n] + gx[dk+n];
        //        gy[dl+n] = ykyl * gy[n] + gy[dk+n];
        //        gz[dl+n] = zkzl * gz[n] + gz[dk+n];
        //    }
        //} }

        // unrolling l
        for (l = 0; l < klmin-1; l+=2, mmax-=2) {
            off = l * dl;
            for (n = off; n < off+dij; ++n) {
                s0x = gx[n+mmax*dk-dk];
                s0y = gy[n+mmax*dk-dk];
                s0z = gz[n+mmax*dk-dk];
                t1x = xkxl * s0x + gx[n+mmax*dk];
                t1y = ykyl * s0y + gy[n+mmax*dk];
                t1z = zkzl * s0z + gz[n+mmax*dk];
                gx[dl+n+mmax*dk-dk] = t1x;
                gy[dl+n+mmax*dk-dk] = t1y;
                gz[dl+n+mmax*dk-dk] = t1z;
                s1x = s0x;
                s1y = s0y;
                s1z = s0z;
                for (k = mmax-2; k >= 0; k--) {
                    s0x = gx[n+k*dk];
                    s0y = gy[n+k*dk];
                    s0z = gz[n+k*dk];
                    t0x = xkxl * s0x + s1x;
                    t0y = ykyl * s0y + s1y;
                    t0z = zkzl * s0z + s1z;
                    gx[dl+n+k*dk] = t0x;
                    gy[dl+n+k*dk] = t0y;
                    gz[dl+n+k*dk] = t0z;
                    gx[dl+dl+n+k*dk] = xkxl * t0x + t1x;
                    gy[dl+dl+n+k*dk] = ykyl * t0y + t1y;
                    gz[dl+dl+n+k*dk] = zkzl * t0z + t1z;
                    s1x = s0x;
                    s1y = s0y;
                    s1z = s0z;
                    t1x = t0x;
                    t1y = t0y;
                    t1z = t0z;
                }
            }
        }

        if (l < klmin) {
            off = l * dl;
            for (n = off; n < off+dij; ++n) {
                s1x = gx[n + mmax*dk];
                s1y = gy[n + mmax*dk];
                s1z = gz[n + mmax*dk];
                for (k = mmax-1; k >= 0; k--) {
                    s0x = gx[n+k*dk];
                    s0y = gy[n+k*dk];
                    s0z = gz[n+k*dk];
                    gx[dl+n+k*dk] = xkxl * s0x + s1x;
                    gy[dl+n+k*dk] = ykyl * s0y + s1y;
                    gz[dl+n+k*dk] = zkzl * s0z + s1z;
                    s1x = s0x;
                    s1y = s0y;
                    s1z = s0z;
                }
            }
        }
    }
}


template <int NROOTS> __device__
static void GINTnabla1i_2e(GINTEnvVars envs, double *f, double *g, double ai2, int li, int lj, int lk){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829
    
    int n, ptr;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    double *gx = g;
    double *gy = g + envs.g_size;
    double *gz = g + envs.g_size * 2;
    double *fx = f;
    double *fy = f + envs.g_size;
    double *fz = f + envs.g_size * 2;
    double *p1x = gx - di;
    double *p1y = gy - di;
    double *p1z = gz - di;
    double *p2x = gx + di;
    double *p2y = gy + di;
    double *p2z = gz + di;
    //printf("%d %d %d\n", di, dj, dk);
    for (int k = 0; k <= lk; k++)
    for (int j = 0; j <= lj; j++) {
        ptr = dj * j + dk * k;
#pragma unroll
        for (n = ptr; n < ptr+NROOTS; ++n){
            fx[n] = ai2 * p2x[n];
            fy[n] = ai2 * p2y[n];
            fz[n] = ai2 * p2z[n];
        }
        ptr += di;
        for (int i = 1; i <= li; i++){
#pragma unroll
            for (n = ptr; n < ptr+NROOTS; ++n) {
                fx[n] = i*p1x[n] + ai2*p2x[n];
                fy[n] = i*p1y[n] + ai2*p2y[n];
                fz[n] = i*p1z[n] + ai2*p2z[n];
            }
            ptr += di;
        }
    }
}


template <int NROOTS> __device__
static void GINTnabla1j_2e(GINTEnvVars envs, double *f, double *g, double aj2, int li, int lj, int lk){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829
    
    int n, ptr;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    double *gx = g;
    double *gy = g + envs.g_size;
    double *gz = g + envs.g_size * 2;
    double *fx = f;
    double *fy = f + envs.g_size;
    double *fz = f + envs.g_size * 2;
    double *p1x = gx - dj;
    double *p1y = gy - dj;
    double *p1z = gz - dj;
    double *p2x = gx + dj;
    double *p2y = gy + dj;
    double *p2z = gz + dj;
    
    for (int k = 0; k <= lk; k++){
        ptr = dk * k;
        for (int i = 0; i <= li; i++) {
#pragma unroll
            for (n = ptr; n < ptr+NROOTS; ++n){
                fx[n] = aj2 * p2x[n];
                fy[n] = aj2 * p2y[n];
                fz[n] = aj2 * p2z[n];
            }
            ptr += di;
        }
        for (int j = 1; j <= lj; j++){
            ptr = dj * j + dk * k;
            for (int i = 0; i <= li; i++){
#pragma unroll
                for (n = ptr; n < ptr+NROOTS; ++n) {
                    fx[n] = j*p1x[n] + aj2*p2x[n];
                    fy[n] = j*p1y[n] + aj2*p2y[n];
                    fz[n] = j*p1z[n] + aj2*p2z[n];
                }
                ptr += di;
            }
        }
    }
}



template <int NROOTS> __device__
static void GINTnabla1k_2e(GINTEnvVars envs, double *f, double *g, double ak2, int li, int lj, int lk){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829
    
    int n, ptr;
    int di = envs.stride_i;
    int dj = envs.stride_j;
    int dk = envs.stride_k;
    double *gx = g;
    double *gy = g + envs.g_size;
    double *gz = g + envs.g_size * 2;
    double *fx = f;
    double *fy = f + envs.g_size;
    double *fz = f + envs.g_size * 2;
    double *p1x = gx - dk;
    double *p1y = gy - dk;
    double *p1z = gz - dk;
    double *p2x = gx + dk;
    double *p2y = gy + dk;
    double *p2z = gz + dk;
    
    for (int j = 0; j <= lj; j++) {
        ptr = dj * j;
        for (int i = 0; i <= li; i++){
#pragma unroll
            for (n = ptr; n < ptr+NROOTS; ++n){
                fx[n] = ak2 * p2x[n];
                fy[n] = ak2 * p2y[n];
                fz[n] = ak2 * p2z[n];
            }
            ptr += di;
        }
    }
    
    for (int k = 1; k <= lk; k++){
        for (int j = 0; j <= lj; j++){
            ptr = dj * j + dk * k;
            for (int i = 0; i <= li; i++){
#pragma unroll
                for (n = ptr; n < ptr+NROOTS; ++n) {
                    fx[n] = k*p1x[n] + ak2*p2x[n];
                    fy[n] = k*p1y[n] + ak2*p2y[n];
                    fz[n] = k*p1z[n] + ak2*p2z[n];
                }
                ptr += di;
            }
        }
    }
}
