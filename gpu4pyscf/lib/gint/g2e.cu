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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "g2e.h"
#include "cint2e.cuh"

// TODO: prime basis into different thread

template <int NROOTS> __device__
static void GINTg0_2e_2d4d(GINTEnvVars envs, double* __restrict__ g, double norm, int ish, int jsh, int ksh, int lsh, int prim_ij, int prim_kl)
{
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double aij = a12[prim_ij];
    double xij = x12[prim_ij];
    double yij = y12[prim_ij];
    double zij = z12[prim_ij];
    double akl = a12[prim_kl];
    double xkl = x12[prim_kl];
    double ykl = y12[prim_kl];
    double zkl = z12[prim_kl];

    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double omega = envs.omega;
    double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    a0 *= theta;

    double eij = e12[prim_ij];
    double ekl = e12[prim_kl];
    double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double uw[NROOTS*2];
    GINTrys_root<NROOTS>(x, uw);
    GINTscale_u<NROOTS>(uw, theta);

    double* __restrict__ u = uw;
    double* __restrict__ w = u + NROOTS;
    double* __restrict__ gx = g;
    double* __restrict__ gy = g + envs.g_size;
    double* __restrict__ gz = g + envs.g_size * 2;

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

__device__
static void GINTg0_int3c2e_shared(GINTEnvVars envs, double* __restrict__ g0,
    const int ish, const int jsh, const int ksh,
    const int prim_ij, const int prim_kl)
{
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    const double aij = a12[prim_ij];
    const double xij = x12[prim_ij];
    const double yij = y12[prim_ij];
    const double zij = z12[prim_ij];
    const double akl = a12[prim_kl];
    const double xkl = x12[prim_kl];
    const double ykl = y12[prim_kl];
    const double zkl = z12[prim_kl];

    const double xijxkl[3] = {xij-xkl, yij-ykl, zij-zkl};
    const double aijkl = aij + akl;
    const double a1 = aij * akl;
    const double a1_aijkl = a1 / aijkl;
    const double omega = envs.omega;
    const double omega2 = omega * omega;
    const double theta = omega > 0.0 ? omega2 / (omega2 + a1_aijkl) : 1.0;
    const double a0 = theta * a1_aijkl;
    const double eij = e12[prim_ij];
    const double ekl = e12[prim_kl];
    const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
    const double x = a0 * (xijxkl[0] * xijxkl[0] + xijxkl[1] * xijxkl[1] + xijxkl[2] * xijxkl[2]);
    const int nrys_roots = envs.nrys_roots;
    double uw[2];
    GINTrys_root(nrys_roots, x, uw);
    double root = uw[0];
    const double weight = uw[1];
    root /= root + 1 - root * theta;

    const int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    const double xi = bas_x[ish];
    const double yi = bas_y[ish];
    const double zi = bas_z[ish];
    const double xijxi[3] = {xij-xi, yij-yi, zij-zi};

    int nmax = envs.li_ceil + envs.lj_ceil;
    const int mmax = envs.lk_ceil + envs.ll_ceil;
    const int ijmin = envs.ijmin;

    const int dm = envs.stride_klmax;
    const int dn = envs.stride_ijmax;
    const int di = envs.stride_ijmax;
    const int dj = envs.stride_ijmin;
    const int dk = envs.stride_klmax;

    const int gsize = envs.g_size;

    __syncthreads();
    for (int i = threadIdx.x; i < nrys_roots; i += blockDim.x) {
        g0[i] = envs.fac;
        g0[i+gsize] = fac;
        g0[i+2*gsize] = weight;
    }
    __syncthreads();

    for (int tx = threadIdx.x; tx < nrys_roots*3; tx += blockDim.x) {
        const int iroot = tx % nrys_roots;
        const int ix = tx / nrys_roots;
        double *gx = g0 + ix * envs.g_size + iroot;
        const double u2 = a0 * root;
        const double tmp4 = .5 / (u2 * aijkl + a1);
        const double b00 = u2 * tmp4;
        const double tmp1 = 2 * b00;
        const double tmp2 = tmp1 * akl;
        const double b10 = b00 + tmp4 * akl;
        const double c00 = xijxi[ix] - tmp2 * xijxkl[ix];

        const double g_0 = gx[0];

        if (nmax > 0) {
            double s0 = g_0;
            double s1 = c00 * s0;
            gx[dn] = s1;
            for (int n = 1; n < nmax; ++n) {
                const double s2 = c00 * s1 + n * b10 * s0;
                gx[(n+1)*dn] = s2;
                s0 = s1; s1 = s2;
            }
        }
        if (mmax > 0) {
            const double tmp3 = tmp1 * aij;
            const double b01 = b00 + tmp4 * aij;
            const double c0p = tmp3 * xijxkl[ix];
            double s0 = g_0;
            double s1 = c0p * s0;
            gx[dm] = s1;
            for (int m = 1; m < mmax; ++m) {
                const double s2 = c0p * s1 + m * b01 * s0;
                gx[(m+1)*dm] = s2;
                s0 = s1; s1 = s2;
            }

            if (nmax > 0) {
                double s0 = gx[dn];
                double s1 = c0p * s0 + b00 * g_0;
                gx[dn+dm] = s1;
                for (int m = 1; m < mmax; ++m) {
                    const double s2 = c0p*s1 + m*b01*s0 + b00*gx[m*dm];
                    gx[dn+(m+1)*dm] = s2;
                    s0 = s1; s1 = s2;
                }
            }
        }

        // gx(irys,m,n+1) = c00(irys)*gx(irys,m,n)
        // + n*b10(irys)*gx(irys,m,n-1)
        // + m*b00(irys)*gx(irys,m-1,n)
        // TODO: run m direction in parallel
        for (int m = 1; m <= mmax; ++m) {
            double s0 = gx[m*dm];
            double s1 = gx[m*dm+ dn];
            const double tmpb0 = m * b00;
            for (int n = 1; n < nmax; ++n) {
                const double s2 = c00*s1 + n*b10*s0 + tmpb0*gx[m*dm+n*dn-dm];
                gx[m*dm+(n+1)*dn] = s2;
                s0 = s1;  s1 = s2;
            }
        }

        if (ijmin > 0) {
            // g(i,j) = rirj * g(i,j-1) +  g(i+1,j-1)
            const double xixj[3] = {xi-bas_x[jsh], yi-bas_y[jsh], zi-bas_z[jsh]};

            // unrolling j
            int j;
            for (j = 0; j < ijmin-1; j+=2, nmax-=2) {
            // run k direction in parallel
            for (int k = 0; k <= mmax; ++k) {
                const int n = k * dk + j * dj;
                double s0 = gx[n+nmax*di-di];
                double t1 = xixj[ix] * s0 + gx[n+nmax*di];
                gx[dj+n+nmax*di-di] = t1;
                double s1 = s0;
                for (int i = nmax-2; i >= 0; i--) {
                    s0 = gx[n+i*di];
                    double t0 = xixj[ix] * s0 + s1;
                    gx[dj+n+i*di] = t0;
                    gx[dj+dj+n+i*di] = xixj[ix] * t0 + t1;
                    s1 = s0; t1 = t0;
                }
            } }

            if (j < ijmin) {
                // run k direction in parallel
                for (int k = 0; k <= mmax; ++k) {
                    const int n = k * dk + j * dj;
                    double s1 = gx[n + nmax*di];
                    for (int i = nmax-1; i >= 0; i--) {
                        double s0 = gx[n+i*di];
                        gx[dj+n+i*di] = xixj[ix] * s0 + s1;
                        s1 = s0;
                    }
                }
            }
        }
    }
    __syncthreads();
}


template <int NROOTS> __device__
static void GINTg0_int3c2e(GINTEnvVars envs, double* __restrict__ g,
    const double norm, const int ish, const int jsh, const int ksh,
    const int prim_ij, const int prim_kl)
{
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    const double aij = a12[prim_ij];
    const double xij = x12[prim_ij];
    const double yij = y12[prim_ij];
    const double zij = z12[prim_ij];
    const double akl = a12[prim_kl];
    const double xkl = x12[prim_kl];
    const double ykl = y12[prim_kl];
    const double zkl = z12[prim_kl];

    const double xijxkl = xij - xkl;
    const double yijykl = yij - ykl;
    const double zijzkl = zij - zkl;
    const double aijkl = aij + akl;
    const double a1 = aij * akl;
    double a0 = a1 / aijkl;
    const double omega = envs.omega;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    a0 *= theta;

    const double eij = e12[prim_ij];
    const double ekl = e12[prim_kl];
    const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
    const double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double uw[NROOTS*2];
    GINTrys_root<NROOTS>(x, uw);
    GINTscale_u<NROOTS>(uw, theta);

    double* __restrict__ u = uw;
    double* __restrict__ w = u + NROOTS;
    double* __restrict__ gx = g;
    double* __restrict__ gy = g + envs.g_size;
    double* __restrict__ gz = g + envs.g_size * 2;

    const int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    const double xi = bas_x[ish];
    const double yi = bas_y[ish];
    const double zi = bas_z[ish];
    const double xijxi = xij - xi;
    const double yijyi = yij - yi;
    const double zijzi = zij - zi;

    int nmax = envs.li_ceil + envs.lj_ceil;
    int mmax = envs.lk_ceil + envs.ll_ceil;
    const int ijmin = envs.ijmin;

    const int dm = envs.stride_klmax;
    const int dn = envs.stride_ijmax;
    const int di = envs.stride_ijmax;
    const int dj = envs.stride_ijmin;
    const int dk = envs.stride_klmax;

    for (int i = 0; i < NROOTS; ++i) {
        gx[i] = norm;
        gy[i] = fac;
        gz[i] = w[i];

        const double u2 = a0 * u[i];
        const double tmp4 = .5 / (u2 * aijkl + a1);
        const double b00 = u2 * tmp4;
        const double tmp1 = 2 * b00;
        const double tmp2 = tmp1 * akl;
        const double b10 = b00 + tmp4 * akl;
        const double c00x = xijxi - tmp2 * xijxkl;
        const double c00y = yijyi - tmp2 * yijykl;
        const double c00z = zijzi - tmp2 * zijzkl;

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
            double s0x = gx[i];
            double s0y = gy[i];
            double s0z = gz[i];
            double s1x = c00x * s0x;
            double s1y = c00y * s0y;
            double s1z = c00z * s0z;
            gx[i+dn] = s1x;
            gy[i+dn] = s1y;
            gz[i+dn] = s1z;
            for (int n = 1; n < nmax; ++n) {
                double s2x = c00x * s1x + n * b10 * s0x;
                double s2y = c00y * s1y + n * b10 * s0y;
                double s2z = c00z * s1z + n * b10 * s0z;
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
            const double tmp3 = tmp1 * aij;
            const double b01 = b00 + tmp4 * aij;
            const double c0px = tmp3 * xijxkl;
            const double c0py = tmp3 * yijykl;
            const double c0pz = tmp3 * zijzkl;
            double s0x = gx[i];
            double s0y = gy[i];
            double s0z = gz[i];
            double s1x = c0px * s0x;
            double s1y = c0py * s0y;
            double s1z = c0pz * s0z;
            gx[i+dm] = s1x;
            gy[i+dm] = s1y;
            gz[i+dm] = s1z;
            for (int m = 1; m < mmax; ++m) {
                double s2x = c0px * s1x + m * b01 * s0x;
                double s2y = c0py * s1y + m * b01 * s0y;
                double s2z = c0pz * s1z + m * b01 * s0z;
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
                double s0x = gx[i+dn];
                double s0y = gy[i+dn];
                double s0z = gz[i+dn];
                double s1x = c0px * s0x + b00 * gx[i];
                double s1y = c0py * s0y + b00 * gy[i];
                double s1z = c0pz * s0z + b00 * gz[i];
                gx[i+dn+dm] = s1x;
                gy[i+dn+dm] = s1y;
                gz[i+dn+dm] = s1z;
                for (int m = 1; m < mmax; ++m) {
                    double s2x = c0px*s1x + m*b01*s0x + b00*gx[i+m*dm];
                    double s2y = c0py*s1y + m*b01*s0y + b00*gy[i+m*dm];
                    double s2z = c0pz*s1z + m*b01*s0z + b00*gz[i+m*dm];
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
        for (int m = 1; m <= mmax; ++m) {
            //for (n = 1; n < nmax; ++n) {
            //    off = m * dm + n * dn;
            //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
            //        gx[j+dn] = c00x[i]*gx[j] +n*b10[i]*gx[j-dn] + m*b00[i]*gx[j-dm];
            //        gy[j+dn] = c00y[i]*gy[j] +n*b10[i]*gy[j-dn] + m*b00[i]*gy[j-dm];
            //        gz[j+dn] = c00z[i]*gz[j] +n*b10[i]*gz[j-dn] + m*b00[i]*gz[j-dm];
            //    }
            //}
            int off = m * dm;
            int j = off + i;
            double s0x = gx[j];
            double s0y = gy[j];
            double s0z = gz[j];
            double s1x = gx[j + dn];
            double s1y = gy[j + dn];
            double s1z = gz[j + dn];
            const double tmpb0 = m * b00;
            for (int n = 1; n < nmax; ++n) {
                double s2x = c00x*s1x + n*b10*s0x + tmpb0*gx[j+n*dn-dm];
                double s2y = c00y*s1y + n*b10*s0y + tmpb0*gy[j+n*dn-dm];
                double s2z = c00z*s1z + n*b10*s0z + tmpb0*gz[j+n*dn-dm];
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
        const double xixj = xi - bas_x[jsh];
        const double yiyj = yi - bas_y[jsh];
        const double zizj = zi - bas_z[jsh];
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
        int j;
        for (j = 0; j < ijmin-1; j+=2, nmax-=2) {
        for (int k = 0; k <= mmax; ++k) {
            int off = k * dk + j * dj;
            for (int n = off; n < off+NROOTS; ++n) {
                double s0x = gx[n+nmax*di-di];
                double s0y = gy[n+nmax*di-di];
                double s0z = gz[n+nmax*di-di];
                double t1x = xixj * s0x + gx[n+nmax*di];
                double t1y = yiyj * s0y + gy[n+nmax*di];
                double t1z = zizj * s0z + gz[n+nmax*di];
                gx[dj+n+nmax*di-di] = t1x;
                gy[dj+n+nmax*di-di] = t1y;
                gz[dj+n+nmax*di-di] = t1z;
                double s1x = s0x;
                double s1y = s0y;
                double s1z = s0z;
                for (int i = nmax-2; i >= 0; i--) {
                    s0x = gx[n+i*di];
                    s0y = gy[n+i*di];
                    s0z = gz[n+i*di];
                    double t0x = xixj * s0x + s1x;
                    double t0y = yiyj * s0y + s1y;
                    double t0z = zizj * s0z + s1z;
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
            for (int k = 0; k <= mmax; ++k) {
                int off = k * dk + j * dj;
                for (int n = off; n < off+NROOTS; ++n) {
                    double s1x = gx[n + nmax*di];
                    double s1y = gy[n + nmax*di];
                    double s1z = gz[n + nmax*di];
                    for (int i = nmax-1; i >= 0; i--) {
                        double s0x = gx[n+i*di];
                        double s0y = gy[n+i*di];
                        double s0z = gz[n+i*di];
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
}


template <int LI, int LJ, int LK> __device__
static void GINTg0_int3c2e(GINTEnvVars envs, double* __restrict__ g,
    const int ish, const int jsh, const int ksh,
    const int prim_ij, const int prim_kl)
{
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    const double aij = a12[prim_ij];
    const double xij = x12[prim_ij];
    const double yij = y12[prim_ij];
    const double zij = z12[prim_ij];
    const double akl = a12[prim_kl];
    const double xkl = x12[prim_kl];
    const double ykl = y12[prim_kl];
    const double zkl = z12[prim_kl];

    const double xijxkl = xij - xkl;
    const double yijykl = yij - ykl;
    const double zijzkl = zij - zkl;
    const double aijkl = aij + akl;
    const double a1 = aij * akl;
    const double a1_aijkl = a1 / aijkl;
    const double omega = envs.omega;
    const double omega2 = omega * omega;
    const double theta = omega > 0.0 ? omega2 / (omega2 + a1_aijkl) : 1.0;
    const double a0 = theta * a1_aijkl;
    const double eij = e12[prim_ij];
    const double ekl = e12[prim_kl];
    const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
    const double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);

    constexpr int NROOTS = (LI + LJ + LK)/2 + 1;
    double uw[NROOTS*2];
    GINTrys_root<NROOTS>(x, uw);
    GINTscale_u<NROOTS>(uw, theta);

    double* __restrict__ u = uw;
    double* __restrict__ w = u + NROOTS;
    double* __restrict__ gx = g;
    double* __restrict__ gy = g + envs.g_size;
    double* __restrict__ gz = g + envs.g_size * 2;

    const int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    const double xi = bas_x[ish];
    const double yi = bas_y[ish];
    const double zi = bas_z[ish];
    const double xk = bas_x[ksh];
    const double yk = bas_y[ksh];
    const double zk = bas_z[ksh];
    const double xijxi = xij - xi;
    const double yijyi = yij - yi;
    const double zijzi = zij - zi;
    const double xklxk = xkl - xk;
    const double yklyk = ykl - yk;
    const double zklzk = zkl - zk;

    constexpr int ijmin = LJ;//envs.ijmin;

    constexpr int di = NROOTS;//envs.stride_ijmax;
    constexpr int dj = NROOTS * (LI+1);//envs.stride_ijmin;
    constexpr int dk = dj * (LJ+1); //envs.stride_klmax;
    constexpr int dm = dk;//envs.stride_klmax;
    constexpr int dn = NROOTS;//envs.stride_ijmax;

    constexpr int NMAX = LI + LJ;
    constexpr int MMAX = LK;

    const double norm = envs.fac;
    for (int i = 0; i < NROOTS; ++i) {
        gx[i] = norm;
        gy[i] = fac;
        gz[i] = w[i];

        const double u2 = a0 * u[i];
        const double tmp4 = .5 / (u2 * aijkl + a1);
        const double b00 = u2 * tmp4;
        const double tmp1 = 2 * b00;
        const double tmp2 = tmp1 * akl;
        const double b10 = b00 + tmp4 * akl;
        const double c00x = xijxi - tmp2 * xijxkl;
        const double c00y = yijyi - tmp2 * yijykl;
        const double c00z = zijzi - tmp2 * zijzkl;

        if (NMAX > 0) {
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
            double s0x = gx[i];
            double s0y = gy[i];
            double s0z = gz[i];
            double s1x = c00x * s0x;
            double s1y = c00y * s0y;
            double s1z = c00z * s0z;
            gx[i+dn] = s1x;
            gy[i+dn] = s1y;
            gz[i+dn] = s1z;
#pragma unroll
            for (int n = 1; n < NMAX; ++n) {
                double s2x = c00x * s1x + n * b10 * s0x;
                double s2y = c00y * s1y + n * b10 * s0y;
                double s2z = c00z * s1z + n * b10 * s0z;
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

        if (MMAX > 0) {
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
            const double tmp3 = tmp1 * aij;
            const double b01 = b00 + tmp4 * aij;
            const double c0px = xklxk + tmp3 * xijxkl;
            const double c0py = yklyk + tmp3 * yijykl;
            const double c0pz = zklzk + tmp3 * zijzkl;
            double s0x = gx[i];
            double s0y = gy[i];
            double s0z = gz[i];
            double s1x = c0px * s0x;
            double s1y = c0py * s0y;
            double s1z = c0pz * s0z;
            gx[i+dm] = s1x;
            gy[i+dm] = s1y;
            gz[i+dm] = s1z;
#pragma unroll
            for (int m = 1; m < MMAX; ++m) {
                const double s2x = c0px * s1x + m * b01 * s0x;
                const double s2y = c0py * s1y + m * b01 * s0y;
                const double s2z = c0pz * s1z + m * b01 * s0z;
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

            if (NMAX > 0) {
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
                double s0x = gx[i+dn];
                double s0y = gy[i+dn];
                double s0z = gz[i+dn];
                double s1x = c0px * s0x + b00 * gx[i];
                double s1y = c0py * s0y + b00 * gy[i];
                double s1z = c0pz * s0z + b00 * gz[i];
                gx[i+dn+dm] = s1x;
                gy[i+dn+dm] = s1y;
                gz[i+dn+dm] = s1z;
#pragma unroll
                for (int m = 1; m < MMAX; ++m) {
                    const double s2x = c0px*s1x + m*b01*s0x + b00*gx[i+m*dm];
                    const double s2y = c0py*s1y + m*b01*s0y + b00*gy[i+m*dm];
                    const double s2z = c0pz*s1z + m*b01*s0z + b00*gz[i+m*dm];
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
#pragma unroll
        for (int m = 1; m <= MMAX; ++m) {
            //for (n = 1; n < nmax; ++n) {
            //    off = m * dm + n * dn;
            //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
            //        gx[j+dn] = c00x[i]*gx[j] +n*b10[i]*gx[j-dn] + m*b00[i]*gx[j-dm];
            //        gy[j+dn] = c00y[i]*gy[j] +n*b10[i]*gy[j-dn] + m*b00[i]*gy[j-dm];
            //        gz[j+dn] = c00z[i]*gz[j] +n*b10[i]*gz[j-dn] + m*b00[i]*gz[j-dm];
            //    }
            //}
            int off = m * dm;
            int j = off + i;
            double s0x = gx[j];
            double s0y = gy[j];
            double s0z = gz[j];
            double s1x = gx[j + dn];
            double s1y = gy[j + dn];
            double s1z = gz[j + dn];
            const double tmpb0 = m * b00;
#pragma unroll
            for (int n = 1; n < NMAX; ++n) {
                const double s2x = c00x*s1x + n*b10*s0x + tmpb0*gx[j+n*dn-dm];
                const double s2y = c00y*s1y + n*b10*s0y + tmpb0*gy[j+n*dn-dm];
                const double s2z = c00z*s1z + n*b10*s0z + tmpb0*gz[j+n*dn-dm];
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
        const double xixj = xi - bas_x[jsh];
        const double yiyj = yi - bas_y[jsh];
        const double zizj = zi - bas_z[jsh];
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
        int nmax = NMAX;
        int j;
        for (j = 0; j < ijmin-1; j+=2, nmax-=2) {
#pragma unroll
        for (int k = 0; k <= MMAX; ++k) {
            int off = k * dk + j * dj;
            for (int n = off; n < off+NROOTS; ++n) {
                double s0x = gx[n+nmax*di-di];
                double s0y = gy[n+nmax*di-di];
                double s0z = gz[n+nmax*di-di];
                double t1x = xixj * s0x + gx[n+nmax*di];
                double t1y = yiyj * s0y + gy[n+nmax*di];
                double t1z = zizj * s0z + gz[n+nmax*di];
                gx[dj+n+nmax*di-di] = t1x;
                gy[dj+n+nmax*di-di] = t1y;
                gz[dj+n+nmax*di-di] = t1z;
                double s1x = s0x;
                double s1y = s0y;
                double s1z = s0z;
                for (int i = nmax-2; i >= 0; i--) {
                    s0x = gx[n+i*di];
                    s0y = gy[n+i*di];
                    s0z = gz[n+i*di];
                    const double t0x = xixj * s0x + s1x;
                    const double t0y = yiyj * s0y + s1y;
                    const double t0z = zizj * s0z + s1z;
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
#pragma unroll
            for (int k = 0; k <= MMAX; ++k) {
                int off = k * dk + j * dj;
                for (int n = off; n < off+NROOTS; ++n) {
                    double s1x = gx[n + nmax*di];
                    double s1y = gy[n + nmax*di];
                    double s1z = gz[n + nmax*di];
                    for (int i = nmax-1; i >= 0; i--) {
                        const double s0x = gx[n+i*di];
                        const double s0y = gy[n+i*di];
                        const double s0z = gz[n+i*di];
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
}

template <int NROOTS> __device__
static void GINTnabla1i_2e(GINTEnvVars envs, double *f, double *g, const double ai2, int li, int lj, int lk){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829

    int n, ptr;
    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    double * __restrict__ gx = g;
    double * __restrict__ gy = g + envs.g_size;
    double * __restrict__ gz = g + envs.g_size * 2;
    double * __restrict__ fx = f;
    double * __restrict__ fy = f + envs.g_size;
    double * __restrict__ fz = f + envs.g_size * 2;
    //double *p1x = gx - di;
    //double *p1y = gy - di;
    //double *p1z = gz - di;
    //double *p2x = gx + di;
    //double *p2y = gy + di;
    //double *p2z = gz + di;

    double g0, g1, g2;

    for (int k = 0; k <= lk; k++)
    for (int j = 0; j <= lj; j++) {
        for (n = 0; n < NROOTS; n++){
            ptr = dj * j + dk * k + n;
            // gx
            g0 = gx[ptr];
            g1 = gx[ptr + di];
            fx[ptr] = ai2 * g1;
            for (int i = 1; i <= li; i++){
                g2 = gx[ptr + (i+1)*di];
                fx[ptr + i*di] = i*g0 + ai2*g2;
                g0 = g1;
                g1 = g2;
            }

            g0 = gy[ptr];
            g1 = gy[ptr + di];
            fy[ptr] = ai2 * g1;
            for (int i = 1; i <= li; i++){
                g2 = gy[ptr + (i+1)*di];
                fy[ptr + i*di] = i*g0 + ai2*g2;
                g0 = g1;
                g1 = g2;
            }

            g0 = gz[ptr];
            g1 = gz[ptr + di];
            fz[ptr] = ai2 * g1;
            for (int i = 1; i <= li; i++){
                g2 = gz[ptr + (i+1)*di];
                fz[ptr + i*di] = i*g0 + ai2*g2;
                g0 = g1;
                g1 = g2;
            }
        }
        /*
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
        */
    }
}

template <int LI, int LJ, int LK, int NROOTS> __device__
static void GINTnabla1i_2e(GINTEnvVars envs, double *f, double *g, const double ai2){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;

    double * __restrict__ gx = g;
    double * __restrict__ gy = g + envs.g_size;
    double * __restrict__ gz = g + envs.g_size * 2;
    double * __restrict__ fx = f;
    double * __restrict__ fy = f + envs.g_size;
    double * __restrict__ fz = f + envs.g_size * 2;
    //double *p1x = gx - di;
    //double *p1y = gy - di;
    //double *p1z = gz - di;
    //double *p2x = gx + di;
    //double *p2y = gy + di;
    //double *p2z = gz + di;

    double gi[LI+2];

    for (int k = 0; k <= LK; k++)
    for (int j = 0; j <= LJ; j++) {
        for (int n = 0; n < NROOTS; n++){
            int ptr = dj * j + dk * k + n;
            // gx
#pragma unroll
            for (int i = 0; i <= LI+1; i++){
                gi[i] = gx[ptr + i*di];
            }
            fx[ptr] = ai2 * gi[1];
#pragma unroll
            for (int i = 1; i <= LI; i++){
                fx[ptr + i*di] = i*gi[i-1] + ai2*gi[i+1];
            }

            // gy
#pragma unroll
            for (int i = 0; i <= LI+1; i++){
                gi[i] = gy[ptr + i*di];
            }
            fy[ptr] = ai2 * gi[1];
#pragma unroll
            for (int i = 1; i <= LI; i++){
                fy[ptr + i*di] = i*gi[i-1] + ai2*gi[i+1];
            }

            // gz
#pragma unroll
            for (int i = 0; i <= LI+1; i++){
                gi[i] = gz[ptr + i*di];
            }
            fz[ptr] = ai2 * gi[1];
#pragma unroll
            for (int i = 1; i <= LI; i++){
                fz[ptr + i*di] = i*gi[i-1] + ai2*gi[i+1];
            }
        }
    }
    /*
    for (int k = 0; k <= LK; k++)
    for (int j = 0; j <= LJ; j++) {
        int ptr = dj * j + dk * k;
#pragma unroll
        for (int n = ptr; n < ptr+NROOTS; ++n){
            fx[n] = ai2 * p2x[n];
            fy[n] = ai2 * p2y[n];
            fz[n] = ai2 * p2z[n];
        }
        ptr += di;
        for (int i = 1; i <= LI; i++){
#pragma unroll
            for (int n = ptr; n < ptr+NROOTS; ++n) {
                fx[n] = i*p1x[n] + ai2*p2x[n];
                fy[n] = i*p1y[n] + ai2*p2y[n];
                fz[n] = i*p1z[n] + ai2*p2z[n];
            }
            ptr += di;
        }
    }
    */
}


template <int NROOTS> __device__
static void GINTnabla1j_2e(GINTEnvVars envs, double *f, double *g, const double aj2, int li, int lj, int lk){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    double * __restrict__ gx = g;
    double * __restrict__ gy = g + envs.g_size;
    double * __restrict__ gz = g + envs.g_size * 2;
    double * __restrict__ fx = f;
    double * __restrict__ fy = f + envs.g_size;
    double * __restrict__ fz = f + envs.g_size * 2;

    double g0, g1, g2;

    for (int k = 0; k <= lk; k++){
    for (int i = 0; i <= li; i++){
        for (int n = 0; n < NROOTS; n++){
            int ptr = di * i + dk * k + n;
            // gx
            g0 = gx[ptr];
            g1 = gx[ptr + dj];
            fx[ptr] = aj2 * g1;
            for (int j = 1; j <= lj; j++){
                g2 = gx[ptr + (j+1)*dj];
                fx[ptr + j*dj] = j*g0 + aj2*g2;
                g0 = g1;
                g1 = g2;
            }

            // gy
            g0 = gy[ptr];
            g1 = gy[ptr + dj];
            fy[ptr] = aj2 * g1;
            for (int j = 1; j <= lj; j++){
                g2 = gy[ptr + (j+1)*dj];
                fy[ptr + j*dj] = j*g0 + aj2*g2;
                g0 = g1;
                g1 = g2;
            }

            // gz
            g0 = gz[ptr];
            g1 = gz[ptr + dj];
            fz[ptr] = aj2 * g1;
            for (int j = 1; j <= lj; j++){
                g2 = gz[ptr + (j+1)*dj];
                fz[ptr + j*dj] = j*g0 + aj2*g2;
                g0 = g1;
                g1 = g2;
            }
        }
    }}
    /*
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
    */
}


template <int LI, int LJ, int LK, int NROOTS> __device__
static void GINTnabla1j_2e(GINTEnvVars envs, double *f, double *g, const double aj2){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    double * __restrict__ gx = g;
    double * __restrict__ gy = g + envs.g_size;
    double * __restrict__ gz = g + envs.g_size * 2;
    double * __restrict__ fx = f;
    double * __restrict__ fy = f + envs.g_size;
    double * __restrict__ fz = f + envs.g_size * 2;

    double gi[LJ+2];

    for (int k = 0; k <= LK; k++){
    for (int i = 0; i <= LI; i++){
        for (int n = 0; n < NROOTS; n++){
            int ptr = di * i + dk * k + n;
            // gx
#pragma unroll
            for (int j = 0; j <= LJ+1; j++){
                gi[j] = gx[ptr + j*dj];
            }
            fx[ptr] = aj2 * gi[1];
#pragma unroll
            for (int j = 1; j <= LJ; j++){
                fx[ptr + j*dj] = j*gi[j-1] + aj2*gi[j+1];
            }

            // gy
#pragma unroll
            for (int j = 0; j <= LJ+1; j++){
                gi[j] = gy[ptr + j*dj];
            }
            fy[ptr] = aj2 * gi[1];
#pragma unroll
            for (int j = 1; j <= LJ; j++){
                fy[ptr + j*dj] = j*gi[j-1] + aj2*gi[j+1];
            }

            // gz
#pragma unroll
            for (int j = 0; j <= LJ+1; j++){
                gi[j] = gz[ptr + j*dj];
            }
            fz[ptr] = aj2 * gi[1];
#pragma unroll
            for (int j = 1; j <= LJ; j++){
                fz[ptr + j*dj] = j*gi[j-1] + aj2*gi[j+1];
            }
        }
    }}
}


template <int NROOTS> __device__
static void GINTnabla1k_2e(GINTEnvVars envs, double *f, double *g, const double ak2, int li, int lj, int lk){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    double * __restrict__ gx = g;
    double * __restrict__ gy = g + envs.g_size;
    double * __restrict__ gz = g + envs.g_size * 2;
    double * __restrict__ fx = f;
    double * __restrict__ fy = f + envs.g_size;
    double * __restrict__ fz = f + envs.g_size * 2;
    //double *p1x = gx - dk;
    //double *p1y = gy - dk;
    //double *p1z = gz - dk;
    //double *p2x = gx + dk;
    //double *p2y = gy + dk;
    //double *p2z = gz + dk;
    double g0, g1, g2;

    for (int j = 0; j <= lj; j++){
    for (int i = 0; i <= li; i++){
        for (int n = 0; n < NROOTS; n++){
            int ptr = di * i + dj * j + n;
            // gx
            g0 = gx[ptr];
            g1 = gx[ptr + dk];
            fx[ptr] = ak2 * g1;
            for (int k = 1; k <= lk; k++){
                g2 = gx[ptr + (k+1)*dk];
                fx[ptr + k*dk] = k*g0 + ak2*g2;
                g0 = g1;
                g1 = g2;
            }

            // gy
            g0 = gy[ptr];
            g1 = gy[ptr + dk];
            fy[ptr] = ak2 * g1;
            for (int k = 1; k <= lk; k++){
                g2 = gy[ptr + (k+1)*dk];
                fy[ptr + k*dk] = k*g0 + ak2*g2;
                g0 = g1;
                g1 = g2;
            }

            // gz
            g0 = gz[ptr];
            g1 = gz[ptr + dk];
            fz[ptr] = ak2 * g1;
            for (int k = 1; k <= lk; k++){
                g2 = gz[ptr + (k+1)*dk];
                fz[ptr + k*dk] = k*g0 + ak2*g2;
                g0 = g1;
                g1 = g2;
            }
        }
    }}
    /*
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
    */
}

template <int LI, int LJ, int LK, int NROOTS> __device__
static void GINTnabla1k_2e(GINTEnvVars envs, double *f, double *g, const double ak2){
    // reference code
    // https://github.com/sunqm/libcint/blob/be610546b935049d0cf65c1099244d45b2ff4e5e/src/g2e.c#L1829

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    double * __restrict__ gx = g;
    double * __restrict__ gy = g + envs.g_size;
    double * __restrict__ gz = g + envs.g_size * 2;
    double * __restrict__ fx = f;
    double * __restrict__ fy = f + envs.g_size;
    double * __restrict__ fz = f + envs.g_size * 2;

    double gi[LK+2];

    for (int j = 0; j <= LJ; j++){
    for (int i = 0; i <= LI; i++){
        for (int n = 0; n < NROOTS; n++){
            int ptr = di * i + dj * j + n;
            // gx
#pragma unroll
            for (int k = 0; k <= LK+1; k++){
                gi[k] = gx[ptr + k*dk];
            }
            fx[ptr] = ak2 * gi[1];
#pragma unroll
            for (int k = 1; k <= LK; k++){
                fx[ptr + k*dk] = k*gi[k-1] + ak2*gi[k+1];
            }

            // gy
#pragma unroll
            for (int k = 0; k <= LK+1; k++){
                gi[k] = gy[ptr + k*dk];
            }
            fy[ptr] = ak2 * gi[1];
#pragma unroll
            for (int k = 1; k <= LK; k++){
                fy[ptr + k*dk] = k*gi[k-1] + ak2*gi[k+1];
            }

            // gz
#pragma unroll
            for (int k = 0; k <= LK+1; k++){
                gi[k] = gz[ptr + k*dk];
            }
            fz[ptr] = ak2 * gi[1];
#pragma unroll
            for (int k = 1; k <= LK; k++){
                fz[ptr + k*dk] = k*gi[k-1] + ak2*gi[k+1];
            }
        }
    }}
}
