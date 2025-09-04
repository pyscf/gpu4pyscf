/*
 * Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include "pbc.cuh"

#if defined(_OPENMP) && _OPENMP >= 201307
    #define PRAGMA_OMP_SIMD _Pragma("omp simd")
#else
    #define PRAGMA_OMP_SIMD
#endif
#define EXPMIN          -700
#define PTR_RADIUS        5
#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

#define BINOMIAL(n, i)  (_BINOMIAL_COEF[_LEN_CART0[n]+i])
const int _LEN_CART[] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};
const int _LEN_CART0[] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120
};
const int _BINOMIAL_COEF[] = {
    1,
    1,   1,
    1,   2,   1,
    1,   3,   3,   1,
    1,   4,   6,   4,   1,
    1,   5,  10,  10,   5,   1,
    1,   6,  15,  20,  15,   6,   1,
    1,   7,  21,  35,  35,  21,   7,   1,
    1,   8,  28,  56,  70,  56,  28,   8,   1,
    1,   9,  36,  84, 126, 126,  84,  36,   9,   1,
    1,  10,  45, 120, 210, 252, 210, 120,  45,  10,   1,
    1,  11,  55, 165, 330, 462, 462, 330, 165,  55,  11,   1,
    1,  12,  66, 220, 495, 792, 924, 792, 495, 220,  66,  12,   1,
    1,  13,  78, 286, 715,1287,1716,1716,1287, 715, 286,  78,  13,   1,
    1,  14,  91, 364,1001,2002,3003,3432,3003,2002,1001, 364,  91,  14,   1,
    1,  15, 105, 455,1365,3003,5005,6435,6435,5005,3003,1365, 455, 105,  15,   1,
};

static inline double fac(const int i) {
    static const double fac_table[] = {
        1.00000000000000000000000000000000E+00,
        1.00000000000000000000000000000000E+00,
        2.00000000000000000000000000000000E+00,
        6.00000000000000000000000000000000E+00,
        2.40000000000000000000000000000000E+01,
        1.20000000000000000000000000000000E+02,
        7.20000000000000000000000000000000E+02,
        5.04000000000000000000000000000000E+03,
        4.03200000000000000000000000000000E+04,
        3.62880000000000000000000000000000E+05,
        3.62880000000000000000000000000000E+06,
        3.99168000000000000000000000000000E+07,
        4.79001600000000000000000000000000E+08,
        6.22702080000000000000000000000000E+09,
        8.71782912000000000000000000000000E+10,
        1.30767436800000000000000000000000E+12,
        2.09227898880000000000000000000000E+13,
        3.55687428096000000000000000000000E+14,
        6.40237370572800000000000000000000E+15,
        1.21645100408832000000000000000000E+17,
        2.43290200817664000000000000000000E+18,
        5.10909421717094400000000000000000E+19,
        1.12400072777760768000000000000000E+21,
        2.58520167388849766400000000000000E+22,
        6.20448401733239439360000000000000E+23,
        1.55112100433309859840000000000000E+25,
        4.03291461126605635584000000000000E+26,
        1.08888694504183521607680000000000E+28,
        3.04888344611713860501504000000000E+29,
        8.84176199373970195454361600000000E+30,
        2.65252859812191058636308480000000E+32, //30!
    };
    return fac_table[i];
}


static inline int modulo(int i, int n)
{
    return (i % n + n) % n;
}


static inline int get_upper_bound(int x0, int nx_per_cell, int ix, int ngridx)
{
    return x0 + MIN(nx_per_cell - x0, ngridx - ix);
}


static inline void vadd(double* c, double* a, double* b)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}


static inline void vsub(double* c, double* a, double* b)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}


static inline double vdot(double* a, double* b)
{
    double out;
    out = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return out;
}


static inline void vscale(double* c, double alpha, double* a)
{
    c[0] = a[0] * alpha;
    c[1] = a[1] * alpha;
    c[2] = a[2] * alpha;
}


static inline double vnorm(double *a)
{
    double norm;
    norm = sqrt(vdot(a, a));
    return norm;
}


static inline void get_lattice_coords(double* r_latt, double* r, double* dh_inv)
{
    const double rx = r[0];
    const double ry = r[1];
    const double rz = r[2];
    r_latt[0] = rx * dh_inv[0] + ry * dh_inv[3] + rz * dh_inv[6];
    r_latt[1] = rx * dh_inv[1] + ry * dh_inv[4] + rz * dh_inv[7];
    r_latt[2] = rx * dh_inv[2] + ry * dh_inv[5] + rz * dh_inv[8];
}


int get_max_num_grid_orth(double* dh, double radius)
{
    double dx = MIN(MIN(dh[0], dh[4]), dh[8]);
    int ngrid = 2 * (int) ceil(radius / dx) + 1;
    return ngrid;
}


void get_grid_spacing(double* dh, double *dh_inv, double* a, double* b, int* mesh)
{
    int i, j;
    for (i = 0; i < 3; i++) {
        const int ni = mesh[i];
        for (j = 0; j < 3; j++) {
            dh[i*3+j] = a[i*3+j] / ni;
            dh_inv[j*3+i] = b[i*3+j] * ni;
        }
    }
}


void dger_(const int* m, const int* n, const double* alpha,
           const double* x, const int* incx, const double* y, const int* incy,
           double* a, const int* lda)
{
    int i, j;
    double temp;

    if (*m <= 0 || *n <= 0 || *alpha == 0.0) {
        return; // Quick return if possible
    }

    for (j = 0; j < *n; j++) {
        if (y[j * (*incy)] != 0.0) {
            temp = *alpha * y[j * (*incy)];
            for (i = 0; i < *m; i++) {
                a[i + j * (*lda)] += x[i * (*incx)] * temp;
            }
        }
    }
}


void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, 
            const double* alpha, const double* a, const int* lda, const double* b, const int* ldb, 
            const double* beta, double* c, const int* ldc)
{
    int i, j, l;
    double temp;
    int nota = (*transa == 'N' || *transa == 'n');
    int notb = (*transb == 'N' || *transb == 'n');

    if (*m <= 0 || *n <= 0 || *k <= 0) {
        return; // Quick return if possible
    }

    // Scale C by beta
    if (*beta == 0.0) {
        for (j = 0; j < *n; j++) {
            for (i = 0; i < *m; i++) {
                c[i + j * (*ldc)] = 0.0;
            }
        }
    } else if (*beta != 1.0) {
        for (j = 0; j < *n; j++) {
            for (i = 0; i < *m; i++) {
                c[i + j * (*ldc)] *= *beta;
            }
        }
    }

    if (*alpha == 0.0)
        return;

    // Perform the matrix multiplication
    if (nota && notb) {
        // C := alpha*A*B + beta*C
        for (j = 0; j < *n; j++) {
            for (l = 0; l < *k; l++) {
                temp = *alpha * b[l + j * (*ldb)];
                for (i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] += temp * a[i + l * (*lda)];
                }
            }
        }
    } else if (!nota && notb) {
        // C := alpha*A**T*B + beta*C
        for (j = 0; j < *n; j++) {
            for (l = 0; l < *k; l++) {
                temp = *alpha * b[l + j * (*ldb)];
                for (i = 0; i < *m; i++) {
                    c[i + j * (*ldc)] += temp * a[l + i * (*lda)];
                }
            }
        }
    } else if (nota && !notb) {
        // C := alpha*A*B**T + beta*C
        for (j = 0; j < *n; j++) {
            for (i = 0; i < *m; i++) {
                temp = 0.0;
                for (l = 0; l < *k; l++) {
                    temp += a[i + l * (*lda)] * b[j + l * (*ldb)];
                }
                c[i + j * (*ldc)] += *alpha * temp;
            }
        }
    } else {
        // C := alpha*A**T*B**T + beta*C
        for (j = 0; j < *n; j++) {
            for (i = 0; i < *m; i++) {
                temp = 0.0;
                for (l = 0; l < *k; l++) {
                    temp += a[l + i * (*lda)] * b[j + l * (*ldb)];
                }
                c[i + j * (*ldc)] += *alpha * temp;
            }
        }
    }
}


static int _orth_components(double *xs_exp, int* bounds, double dx, double radius,
                            double xi, double xj, double ai, double aj,
                            int nx_per_cell, int topl, double *cache)
{
    double aij = ai + aj;
    double xij = (ai * xi + aj * xj) / aij;
    int x0_latt = (int) floor((xij - radius) / dx);
    int x1_latt = (int) ceil((xij + radius) / dx);
    int xij_latt = (int) rint(xij / dx);
    xij_latt = MAX(xij_latt, x0_latt);
    xij_latt = MIN(xij_latt, x1_latt);
    bounds[0] = x0_latt;
    bounds[1] = x1_latt;
    int ngridx = x1_latt - x0_latt;

    double base_x = dx * xij_latt;
    double x0xij = base_x - xij;
    double _x0x0 = -aij * x0xij * x0xij;
    if (_x0x0 < EXPMIN) {
        return 0;
    }

    double *gridx = cache;
    double *xs_all = xs_exp;
    if (ngridx > nx_per_cell) {
        xs_all = gridx + ngridx;
    }

    double _dxdx = -aij * dx * dx;
    double _x0dx = -2 * aij * x0xij * dx;
    double exp_dxdx = exp(_dxdx);
    double exp_2dxdx = exp_dxdx * exp_dxdx;
    double exp_x0dx = exp(_dxdx + _x0dx);
    double exp_x0x0_cache = exp(_x0x0);
    double exp_x0x0 = exp_x0x0_cache;

    int i;
    int istart = xij_latt - x0_latt;
    for (i = istart; i < ngridx; i++) {
        xs_all[i] = exp_x0x0;
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
    }

    exp_x0dx = exp(_dxdx - _x0dx);
    exp_x0x0 = exp_x0x0_cache;
    for (i = istart-1; i >= 0; i--) {
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
        xs_all[i] = exp_x0x0;
    }

    if (topl > 0) {
        double x0xi = x0_latt * dx - xi;
        for (i = 0; i < ngridx; i++) {
            gridx[i] = x0xi + i * dx;
        }
        int l;
        double *px0;
        for (l = 1; l <= topl; l++) {
            px0 = xs_all + (l-1) * ngridx;
            for (i = 0; i < ngridx; i++) {
                px0[ngridx+i] = px0[i] * gridx[i];
            }
        }
    }

    // add up contributions from all images to the reference image
    if (ngridx > nx_per_cell) {
        memset(xs_exp, 0, (topl+1)*nx_per_cell*sizeof(double));
        int ix, l, lb, ub, size_x;
        for (ix = 0; ix < ngridx;) {
            lb = modulo(ix + x0_latt, nx_per_cell);
            ub = get_upper_bound(lb, nx_per_cell, ix, ngridx);
            size_x = ub - lb;
            double* __restrict ptr_xs_exp = xs_exp + lb;
            double* __restrict ptr_xs_all = xs_all + ix;
            for (l = 0; l <= topl; l++) {
                //#pragma omp simd
                PRAGMA_OMP_SIMD
                for (i = 0; i < size_x; i++) {
                    ptr_xs_exp[i] += ptr_xs_all[i];
                }
                ptr_xs_exp += nx_per_cell;
                ptr_xs_all += ngridx;
            }
            ix += size_x;
        }

        bounds[0] = 0;
        bounds[1] = nx_per_cell;
        ngridx = nx_per_cell;
    }
    return ngridx;
}


int init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                   int *grid_slice, double* dh, int* mesh, int topl, double radius,
                   double ai, double aj, double *ri, double *rj, double *cache)
{
    int l1 = topl + 1;
    *xs_exp = cache;
    *ys_exp = *xs_exp + l1 * mesh[0];
    *zs_exp = *ys_exp + l1 * mesh[1];
    int data_size = l1 * (mesh[0] + mesh[1] + mesh[2]);
    cache += data_size;

    int ngridx = _orth_components(*xs_exp, grid_slice, dh[0], radius,
                                  ri[0], rj[0], ai, aj, mesh[0], topl, cache);
    if (ngridx == 0) {
        return 0;
    }

    int ngridy = _orth_components(*ys_exp, grid_slice+2, dh[4], radius,
                                  ri[1], rj[1], ai, aj, mesh[1], topl, cache);
    if (ngridy == 0) {
        return 0;
    }

    int ngridz = _orth_components(*zs_exp, grid_slice+4, dh[8], radius,
                                  ri[2], rj[2], ai, aj, mesh[2], topl, cache);
    if (ngridz == 0) {
        return 0;
    }

    return data_size;
}


static void _nonorth_bounds_tight(int* bounds, int* rp_latt, double* roff,
                                  double* rp, double* dh, double* dh_inv, double radius)
{
    int i, j, ia;
    double r_frac[3];

    get_lattice_coords(r_frac, rp, dh_inv);
    rp_latt[0] = (int)rint(r_frac[0]);
    rp_latt[1] = (int)rint(r_frac[1]);
    rp_latt[2] = (int)rint(r_frac[2]);

    roff[0] = rp_latt[0] - r_frac[0];
    roff[1] = rp_latt[1] - r_frac[1];
    roff[2] = rp_latt[2] - r_frac[2];

    for (i = 0; i < 6; i += 2) {
        bounds[i] = INT_MAX;
        bounds[i + 1] = INT_MIN;
    }

    double a_norm[3], e[3][3];
    for (i = 0; i < 3; i++) {
        double *a = dh + i * 3;
        a_norm[i] = vnorm(a);
        vscale(e[i], 1. / a_norm[i], a);
    }

    const int idx[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (i = 0; i < 3; i++) {
        int i1 = idx[i][0];
        int i2 = idx[i][1];
        double *a1 = dh + i1 * 3;
        double *a2 = dh + i2 * 3;
        double theta = .5 * acos(vdot(a1, a2) / (a_norm[i1] * a_norm[i2]));
        double r1 = radius / sin(theta);
        double r2 = r1 * cos(theta);

        double *e1 = e[i1], *e2 = e[i2];
        double e12[3];
        vadd(e12, e1, e2);
        double e12_norm = vnorm(e12);
        vscale(e12, 1. / e12_norm * r1, e12);

        double rp_plus_e12[3], rp_minus_e12[3];
        vadd(rp_plus_e12, rp, e12);
        vsub(rp_minus_e12, rp, e12);

        double e1_times_r2[3], e2_times_r2[3];
        vscale(e1_times_r2, r2, e1);
        vscale(e2_times_r2, r2, e2);

        //four points where the polygon is tangent to the circle
        double c[4][3];
        vsub(c[0], rp_plus_e12, e2_times_r2);
        vsub(c[1], rp_plus_e12, e1_times_r2);
        vadd(c[2], rp_minus_e12, e2_times_r2);
        vadd(c[3], rp_minus_e12, e1_times_r2);

        for (j = 0; j < 4; j++) {
            get_lattice_coords(r_frac, c[j], dh_inv);
            for (ia = 0; ia < 3; ia++) {
                bounds[ia * 2] = MIN(bounds[ia * 2], (int)floor(r_frac[ia]));
                bounds[ia * 2 + 1] = MAX(bounds[ia * 2 + 1], (int)ceil(r_frac[ia]));
            }
        }
    }
}


int get_max_num_grid_nonorth_tight(double*dh, double* dh_inv, double radius)
{
    int bounds[6];
    int rp_latt[3];
    double roff[3];
    double rp[3] = {0};

    _nonorth_bounds_tight(bounds, rp_latt, roff, rp, dh, dh_inv, radius);

    int nx = bounds[1] - bounds[0];
    int ny = bounds[3] - bounds[2];
    int nz = bounds[5] - bounds[4];
    int nmax = MAX(MAX(nx, ny), nz) + 1;
    return nmax;
}


static void _poly_exp(double *xs_all, int* bounds, double dx,
                      double xi, double xoff, int xp_latt, double ap,
                      int topl, double *cache)
{
    int x0_latt = bounds[0];
    int x1_latt = bounds[1];
    int ngridx = x1_latt - x0_latt;

    double _x0x0 = -ap * xoff * xoff;
    if (_x0x0 < EXPMIN) {
        return;
    }

    double _dxdx = -ap * dx * dx;
    double _x0dx = -2 * ap * xoff * dx;
    double exp_dxdx = exp(_dxdx);
    double exp_2dxdx = exp_dxdx * exp_dxdx;
    double exp_x0dx = exp(_dxdx + _x0dx);
    double exp_x0x0_cache = exp(_x0x0);
    double exp_x0x0 = exp_x0x0_cache;

    int i;
    int istart = xp_latt - x0_latt;
    for (i = istart; i < ngridx; i++) {
        xs_all[i] = exp_x0x0;
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
    }

    exp_x0dx = exp(_dxdx - _x0dx);
    exp_x0x0 = exp_x0x0_cache;
    for (i = istart-1; i >= 0; i--) {
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
        xs_all[i] = exp_x0x0;
    }

    if (topl > 0) {
        double *gridx = cache;
        double x0xi = x0_latt * dx - xi;
        for (i = 0; i < ngridx; i++) {
            gridx[i] = x0xi + i * dx;
        }
        int l;
        double *px0 = xs_all;
        double *px1 = px0 + ngridx;
        for (l = 1; l <= topl; l++) {
            for (i = 0; i < ngridx; i++) {
                px1[i] = px0[i] * gridx[i];
            }
            px0 += ngridx;
            px1 += ngridx;
        }
    }
}


static void _nonorth_exp_i(double* out, int* bounds, int i0, double alpha)
{
    int i;
    int istart = bounds[0];
    int iend = bounds[1];
    const double c_exp = exp(alpha);

    out[0] = exp(alpha * (istart - i0));
    for (i = 1; i < (iend - istart); i++) {
        out[i] = out[i-1] * c_exp;
    }
}


static void _nonorth_exp_ij(double* out, int* bounds_i, int* bounds_j,
                            int i0, int j0, double alpha)
{
    int i, j;
    const int istart = bounds_i[0];
    const int iend = bounds_i[1];
    const int ni = iend - istart;
    const int jstart = bounds_j[0];
    const int jend = bounds_j[1];
    const int nj = jend - jstart;

    double c_exp = exp(alpha);
    double c_exp_i = exp(alpha * (istart - i0));
    double ctmp, c_exp_j;
    double *pout;

    ctmp = c_exp_j = 1.;
    for (j = j0 - jstart; j < nj; j++) {
        pout = out + ni * j;
        double ctmp1 = ctmp;
        for (i = 0; i < ni; i++) {
            pout[i] *= ctmp1;
            ctmp1 *= c_exp_j;
        }
        ctmp *= c_exp_i;
        c_exp_j *= c_exp;
    }

    c_exp_i = 1. / c_exp_i;
    c_exp = 1. / c_exp;
    ctmp = c_exp_i;
    c_exp_j = c_exp;
    for (j = j0 - jstart - 1; j >= 0; j--) {
        pout = out + ni * j;
        double ctmp1 = ctmp;
        for (i = 0; i < ni; i++) {
            pout[i] *= ctmp1;
            ctmp1 *= c_exp_j;
        }
        ctmp *= c_exp_i;
        c_exp_j *= c_exp;
    }
}


static void _nonorth_exp_correction(double* exp_corr, int* bounds,
                                    double* dh, double* roff, int* rp_latt,
                                    double ap, double* cache)
{
    const int idx[3][2] = {{1, 0}, {2, 1}, {2, 0}};
    const double c[3] = {
        //a1 * a2
        -2. * ap * (dh[0] * dh[3] + dh[1] * dh[4] + dh[2] * dh[5]),
        //a2 * a3
        -2. * ap * (dh[6] * dh[3] + dh[7] * dh[4] + dh[8] * dh[5]),
        //a3 * a1
        -2. * ap * (dh[0] * dh[6] + dh[1] * dh[7] + dh[2] * dh[8])
    };

    const int ng[3] = {
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    };
    const int nmax = MAX(MAX(ng[0], ng[1]), ng[2]);
    const int I1 = 1;
    double *exp1 = cache;
    double *exp2 = exp1 + nmax;

    int i;
    for (i = 0; i < 3; i++) {
        int i1 = idx[i][0];
        int i2 = idx[i][1];
        double c_exp = exp(c[i] * roff[i1] * roff[i2]);

        _nonorth_exp_i(exp1, bounds+i1*2, rp_latt[i1], c[i] * roff[i2]);
        _nonorth_exp_i(exp2, bounds+i2*2, rp_latt[i2], c[i] * roff[i1]);

        int n1 = ng[i1];
        int n2 = ng[i2];
        const size_t n12 = (size_t)n1 * n2;
        memset(exp_corr, 0, n12 * sizeof(double));
        dger_(&n1, &n2, &c_exp, exp1, &I1, exp2, &I1, exp_corr, &n1);

        _nonorth_exp_ij(exp_corr, bounds+i1*2, bounds+i2*2,
                        rp_latt[i1], rp_latt[i2], c[i]);
        exp_corr += n12;
    }
}


size_t init_nonorth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                         double **exp_corr, int *bounds,
                         double* dh, double* dh_inv,
                         int* mesh, int topl, double radius,
                         double ai, double aj, double *ri, double *rj, double *cache)
{
    int l1 = topl + 1;
    int rp_latt[3];
    double ap = ai + aj;
    double bas2[3], rp[3], roff[3], ri_frac[3];

    bas2[0] = dh[0] * dh[0] + dh[1] * dh[1] + dh[2] * dh[2];
    bas2[1] = dh[3] * dh[3] + dh[4] * dh[4] + dh[5] * dh[5];
    bas2[2] = dh[6] * dh[6] + dh[7] * dh[7] + dh[8] * dh[8];

    get_lattice_coords(ri_frac, ri, dh_inv);

    rp[0] = (ai * ri[0] + aj * rj[0]) / ap;
    rp[1] = (ai * ri[1] + aj * rj[1]) / ap;
    rp[2] = (ai * ri[2] + aj * rj[2]) / ap;

    //_nonorth_bounds(bounds, rp_latt, roff, rp, dh_inv, radius);
    _nonorth_bounds_tight(bounds, rp_latt, roff, rp, dh, dh_inv, radius);

    *xs_exp = cache;
    int ngridx = bounds[1] - bounds[0];
    if (ngridx <= 0) {
        return 0;
    }
    cache += l1 * ngridx;
    _poly_exp(*xs_exp, bounds, 1., ri_frac[0],
              roff[0], rp_latt[0], ap * bas2[0], topl, cache);

    *ys_exp = cache;
    int ngridy = bounds[3] - bounds[2];
    if (ngridy <= 0) {
        return 0;
    }
    cache += l1 * ngridy;
    _poly_exp(*ys_exp, bounds+2, 1., ri_frac[1],
              roff[1], rp_latt[1], ap * bas2[1], topl, cache);

    *zs_exp = cache;
    int ngridz = bounds[5] - bounds[4];
    if (ngridz <= 0) {
        return 0;
    }
    cache += l1 * ngridz;
    _poly_exp(*zs_exp, bounds+4, 1., ri_frac[2],
              roff[2], rp_latt[2], ap * bas2[2], topl, cache);

    *exp_corr = cache;
    size_t exp_corr_size = (size_t)ngridx * ngridy
                         + (size_t)ngridy * ngridz
                         + (size_t)ngridz * ngridx;
    cache += exp_corr_size;
    _nonorth_exp_correction(*exp_corr, bounds,
                            dh, roff, rp_latt,
                            ap, cache);

    size_t data_size = l1 * (ngridx + ngridy + ngridz) + exp_corr_size;
    return data_size;
}


void get_dm_to_dm_xyz_coeff(double* coeff, double* rij, int lmax, double* cache)
{
    int l1 = lmax + 1;
    int l, lx;

    double *rx_pow = cache;
    double *ry_pow = rx_pow + l1;
    double *rz_pow = ry_pow + l1;

    rx_pow[0] = 1.0;
    ry_pow[0] = 1.0;
    rz_pow[0] = 1.0;
    for (lx = 1; lx <= lmax; lx++) {
        rx_pow[lx] = rx_pow[lx-1] * rij[0];
        ry_pow[lx] = ry_pow[lx-1] * rij[1];
        rz_pow[lx] = rz_pow[lx-1] * rij[2];
    }

    int dj = _LEN_CART[lmax];
    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    for (l = 0; l <= lmax; l++){
        for (lx = 0; lx <= l; lx++) {
            pcx[lx] = BINOMIAL(l, lx) * rx_pow[l-lx];
            pcy[lx] = BINOMIAL(l, lx) * ry_pow[l-lx];
            pcz[lx] = BINOMIAL(l, lx) * rz_pow[l-lx];
        }
        pcx += l+1;
        pcy += l+1;
        pcz += l+1;
    }
}


void dm_ijk_to_dm_xyz(double* dm_ijk, double* dm_xyz, double* dh, int topl)
{
    if (topl == 0) {
        dm_xyz[0] = dm_ijk[0];
        return;
    }

    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    double dh_pow[l1][9];
    int i, l;
    for (i = 0; i < 9; i++) {
        dh_pow[0][i] = 1.;
        for (l = 1; l <= topl; l++) {
            dh_pow[l][i] = dh_pow[l - 1][i] * dh[i];
        }
    }

    int lx, ly, lz;
    int ix, jx, kx;
    int iy, jy, ky;
    int iz, jz, kz;
    for (lx = 0; lx <= topl; lx++) {
    for (ix = 0; ix <= lx; ix++) {
    for (jx = 0; jx <= lx-ix; jx++) {
        kx = lx - ix - jx;
        double cx = dh_pow[ix][0] * dh_pow[jx][3] * dh_pow[kx][6]
                  * fac(lx) / (fac(ix) * fac(jx) * fac(kx));

        for (ly = 0; ly <= topl-lx; ly++) {
        for (iy = 0; iy <= ly; iy++) {
        for (jy = 0; jy <= ly-iy; jy++) {
            ky = ly - iy - jy;
            double cxy = cx * dh_pow[iy][1] * dh_pow[jy][4] * dh_pow[ky][7]
                       * fac(ly) / (fac(iy) * fac(jy) * fac(ky)); 

            for (lz = 0; lz <= topl-lx-ly ; lz++) {
                double *ptr_dm_xyz = dm_xyz + lx*l1l1+ly*l1+lz;
            for (iz = 0; iz <= lz; iz++) {
            for (jz = 0; jz <= lz-iz; jz++) {
                kz = lz - iz - jz;
                double cxyz = cxy * dh_pow[iz][2] * dh_pow[jz][5] * dh_pow[kz][8]
                            * fac(lz) / (fac(iz) * fac(jz) * fac(kz));

                int li = ix + iy + iz;
                int lj = jx + jy + jz;
                int lk = kx + ky + kz;

                *ptr_dm_xyz += dm_ijk[li*l1l1+lj*l1+lk] * cxyz;
            }}}
        }}}
    }}}
}


void dgemm_wrapper(const char transa, const char transb,
                   const int m, const int n, const int k,
                   const double alpha, const double* a, const int lda,
                   const double* b, const int ldb,
                   const double beta, double* c, const int ldc)
{
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}


static void integrate_submesh(double* out, double* weights,
                              double* xs_exp, double* ys_exp, double* zs_exp,
                              double fac, int topl,
                              int* mesh_lb, int* mesh_ub, int* submesh_lb,
                              const int* mesh, const int* submesh, double* cache)
{
    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    const int x0 = mesh_lb[0];
    const int y0 = mesh_lb[1];
    const int z0 = mesh_lb[2];

    const int nx = mesh_ub[0] - x0;
    const int ny = mesh_ub[1] - y0;
    const int nz = mesh_ub[2] - z0;

    const int x0_sub = submesh_lb[0];
    const int y0_sub = submesh_lb[1];
    const int z0_sub = submesh_lb[2];

    const size_t mesh_yz = ((size_t) mesh[1]) * mesh[2];

    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D0 = 0;
    const double D1 = 1;

    double *lzlyx = cache;
    double *zly = lzlyx + l1l1 * nx;
    double *ptr_weights = weights + x0 * mesh_yz + y0 * mesh[2] + z0;

    int ix;
    for (ix = 0; ix < nx; ix++) {
        dgemm_wrapper(TRANS_N, TRANS_N, nz, l1, ny,
                      D1, ptr_weights, mesh[2], ys_exp+y0_sub, submesh[1],
                      D0, zly, nz);
        dgemm_wrapper(TRANS_T, TRANS_N, l1, l1, nz,
                      D1, zs_exp+z0_sub, submesh[2], zly, nz,
                      D0, lzlyx+l1l1*ix, l1);
        ptr_weights += mesh_yz;
    }
    dgemm_wrapper(TRANS_N, TRANS_N, l1l1, l1, nx,
                  fac, lzlyx, l1l1, xs_exp+x0_sub, submesh[0],
                  D1, out, l1l1);
}


static void integrate_submesh_nonorth(
        double* out, double* weights,
        double* xs_exp, double* ys_exp, double* zs_exp, double* exp_corr,
        double fac, int topl, int* mesh_lb, int* mesh_ub, int* submesh_lb,
        const int* mesh, const int* submesh, double* cache)
{
    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    const int x0 = mesh_lb[0];
    const int y0 = mesh_lb[1];
    const int z0 = mesh_lb[2];

    const int nx = mesh_ub[0] - x0;
    const int ny = mesh_ub[1] - y0;
    const int nz = mesh_ub[2] - z0;
    const size_t nxy = (size_t)nx * ny;
    const size_t nyz = (size_t)ny * nz;

    const int x0_sub = submesh_lb[0];
    const int y0_sub = submesh_lb[1];
    const int z0_sub = submesh_lb[2];

    const size_t mesh_yz = (size_t)mesh[1] * mesh[2];

    const size_t submesh_xy = (size_t)submesh[0] * submesh[1];
    const size_t submesh_yz = (size_t)submesh[1] * submesh[2];

    double *exp_corr_ij = exp_corr;
    double *exp_corr_jk = exp_corr_ij + submesh_xy;
    double *exp_corr_ik = exp_corr_jk + submesh_yz;

    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D0 = 0;
    const double D1 = 1;

    weights += x0 * mesh_yz + y0 * mesh[2] + z0;
    exp_corr_ij += x0_sub * submesh[1] + y0_sub;
    exp_corr_jk += y0_sub * submesh[2] + z0_sub;
    exp_corr_ik += x0_sub * submesh[2] + z0_sub;

    xs_exp += x0_sub;
    ys_exp += y0_sub;
    zs_exp += z0_sub;

    int i, j, k;
    double *rho = cache;
    double *prho = rho;
    for (i = 0; i < nx; i++) {
        double *pw = weights + i * mesh_yz;
        double *ptmp = exp_corr_jk;
        for (j = 0; j < ny; j++) {
            double tmp = exp_corr_ij[j];
            for (k = 0; k < nz; k++) {
                prho[k] = pw[k] * tmp * ptmp[k] * exp_corr_ik[k];
            }
            pw += mesh[2];
            prho += nz;
            ptmp += submesh[2];
        }
        exp_corr_ij += submesh[1];
        exp_corr_ik += submesh[2];
    }

    double *xyr = rho + nx * nyz;
    dgemm_wrapper(TRANS_T, TRANS_N, l1, nxy, nz,
                  D1, zs_exp, submesh[2], rho, nz,
                  D0, xyr, l1);

    const int l1y = l1 * ny;
    double *xqr = xyr + l1 * nxy;
    for (i = 0; i < nx; i++) {
        dgemm_wrapper(TRANS_N, TRANS_N, l1, l1, ny,
                      D1, xyr + i * l1y, l1, ys_exp, submesh[1],
                      D0, xqr + i * l1l1, l1);
    }

    dgemm_wrapper(TRANS_N, TRANS_N, l1l1, l1, nx,
                  fac, xqr, l1l1, xs_exp, submesh[0],
                  D1, out, l1l1);
}


static void _orth_ints(double *out, double *weights, int topl, double fac,
                       double *xs_exp, double *ys_exp, double *zs_exp,
                       int *grid_slice, int *mesh, double *cache)
{// NOTE: out is accumulated
    const int nx0 = grid_slice[0];
    const int nx1 = grid_slice[1];
    const int ny0 = grid_slice[2];
    const int ny1 = grid_slice[3];
    const int nz0 = grid_slice[4];
    const int nz1 = grid_slice[5];
    const int ngridx = nx1 - nx0;
    const int ngridy = ny1 - ny0;
    const int ngridz = nz1 - nz0;
    if (ngridx == 0 || ngridy == 0 || ngridz == 0) {
        return;
    }

    const int submesh[3] = {ngridx, ngridy, ngridz};
    int lb[3], ub[3];
    int ix, iy, iz;
    for (ix = 0; ix < ngridx;) {
        lb[0] = modulo(ix + nx0, mesh[0]);
        ub[0] = get_upper_bound(lb[0], mesh[0], ix, ngridx);
        for (iy = 0; iy < ngridy;) {
            lb[1] = modulo(iy + ny0, mesh[1]);
            ub[1] = get_upper_bound(lb[1], mesh[1], iy, ngridy);
            for (iz = 0; iz < ngridz;) {
                lb[2] = modulo(iz + nz0, mesh[2]);
                ub[2] = get_upper_bound(lb[2], mesh[2], iz, ngridz);
                int lb_sub[3] = {ix, iy, iz};
                integrate_submesh(out, weights, xs_exp, ys_exp, zs_exp, fac, topl,
                                  lb, ub, lb_sub, mesh, submesh, cache);
                iz += ub[2] - lb[2];
            }
            iy += ub[1] - lb[1];
        }
        ix += ub[0] - lb[0];
    }
}


static void _nonorth_ints(double *out, double *weights, int topl, double fac,
                          double *xs_exp, double *ys_exp, double *zs_exp, double *exp_corr,
                          int *grid_slice, int *mesh, double *cache)
{
    const int nx = mesh[0];
    const int ny = mesh[1];
    const int nz = mesh[2];
    const int nx0 = grid_slice[0];
    const int ny0 = grid_slice[2];
    const int nz0 = grid_slice[4];
    const int ngridx = grid_slice[1] - nx0;
    const int ngridy = grid_slice[3] - ny0;
    const int ngridz = grid_slice[5] - nz0;
    if (ngridx == 0 || ngridy == 0 || ngridz == 0) {
        return;
    }

    const int submesh[3] = {ngridx, ngridy, ngridz};
    int lb[3], ub[3];
    int ix, iy, iz;
    for (ix = 0; ix < ngridx;) {
        lb[0] = modulo(ix + nx0, nx);
        ub[0] = get_upper_bound(lb[0], nx, ix, ngridx);
        for (iy = 0; iy < ngridy;) {
            lb[1] = modulo(iy + ny0, ny);
            ub[1] = get_upper_bound(lb[1], ny, iy, ngridy);
            for (iz = 0; iz < ngridz;) {
                lb[2] = modulo(iz + nz0, nz);
                ub[2] = get_upper_bound(lb[2], nz, iz, ngridz);
                int lb_sub[3] = {ix, iy, iz};
                integrate_submesh_nonorth(out, weights, xs_exp, ys_exp, zs_exp, exp_corr,
                                          fac, topl, lb, ub, lb_sub, mesh, submesh, cache);
                iz += ub[2] - lb[2];
            }
            iy += ub[1] - lb[1];
        }
        ix += ub[0] - lb[0];
    }
}


#define VRHO_LOOP_IP1(X, Y, Z) \
    int lx, ly, lz; \
    int jx, jy, jz; \
    int l##X##_i_m1 = l##X##_i - 1; \
    int l##X##_i_p1 = l##X##_i + 1; \
    double cx, cy, cz, cfac; \
    double fac_i = -2.0 * ai; \
    for (j##Y = 0; j##Y <= l##Y##_j; j##Y++) { \
        c##Y = pc##Y[j##Y+_LEN_CART0[l##Y##_j]]; \
        l##Y = l##Y##_i + j##Y; \
        for (j##Z = 0; j##Z <= l##Z##_j; j##Z++) { \
            c##Z = pc##Z[j##Z+_LEN_CART0[l##Z##_j]]; \
            l##Z = l##Z##_i + j##Z; \
            cfac = c##Y * c##Z; \
            for (j##X = 0; j##X <= l##X##_j; j##X++) { \
                if (l##X##_i > 0) { \
                    c##X = pc##X[j##X+_LEN_CART0[l##X##_j]] * l##X##_i; \
                    l##X = l##X##_i_m1 + j##X; \
                    pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
                } \
                c##X = pc##X[j##X+_LEN_CART0[l##X##_j]] * fac_i; \
                l##X = l##X##_i_p1 + j##X; \
                pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
            } \
        } \
    }


static void _vrho_loop_ip1_x(double* pv1, double* v1_xyz,
                             double* pcx, double* pcy, double* pcz,
                             double ai, double aj,
                             int lx_i, int ly_i, int lz_i,
                             int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VRHO_LOOP_IP1(x,y,z);
}


static void _vrho_loop_ip1_y(double* pv1, double* v1_xyz,
                             double* pcx, double* pcy, double* pcz,
                             double ai, double aj,
                             int lx_i, int ly_i, int lz_i,
                             int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VRHO_LOOP_IP1(y,x,z);
}


static void _vrho_loop_ip1_z(double* pv1, double* v1_xyz,
                             double* pcx, double* pcy, double* pcz,
                             double ai, double aj,
                             int lx_i, int ly_i, int lz_i,
                             int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VRHO_LOOP_IP1(z,x,y);
}


static void _v1_xyz_to_v1(void (*_v1_loop)(), double* v1_xyz, double* v1,
                          int li, int lj, double ai, double aj,
                          double* ri, double* rj, double* cache)
{
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int l1 = li + lj + 2;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj+1];
    cache += 3 * dj;

    get_dm_to_dm_xyz_coeff(coeff, rij, lj+1, cache);

    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    double *pv1 = v1;
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    _v1_loop(pv1, v1_xyz, pcx, pcy, pcz, ai, aj,
                             lx_i, ly_i, lz_i, lx_j, ly_j, lz_j, l1, l1l1);
                    pv1 += 1;
                }
            }
        }
    }
}


int eval_mat_lda_orth_ip1(double *weights, double *out, int comp,
                          int li, int lj, double ai, double aj,
                          double *ri, double *rj, double fac, double cutoff,
                          int dimension, double* dh, double *dh_inv,
                          int *mesh, double *cache)
{
        int dij = _LEN_CART[li] * _LEN_CART[lj];
        int topl = li + lj + 1;
        int l1 = topl+1;
        int l1l1l1 = l1*l1*l1;
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = init_orth_data(&xs_exp, &ys_exp, &zs_exp,
                                       grid_slice, dh, mesh, topl, cutoff,
                                       ai, aj, ri, rj, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        double *mat_xyz = cache;
        cache += l1l1l1;
        double *pout_x = out;
        double *pout_y = pout_x + dij;
        double *pout_z = pout_y + dij;

        memset(mat_xyz, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_xyz, weights, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_x, mat_xyz, pout_x, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_y, mat_xyz, pout_y, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_z, mat_xyz, pout_z, li, lj, ai, aj, ri, rj, cache);
        return 1;
}


int eval_mat_lda_nonorth_ip1(double *weights, double *out, int comp,
                             int li, int lj, double ai, double aj,
                             double *ri, double *rj, double fac, double cutoff,
                             int dimension, double* dh, double *dh_inv,
                             int *mesh, double *cache)
{
    int dij = _LEN_CART[li] * _LEN_CART[lj];
    int topl = li + lj + 1;
    int l1 = topl+1;
    int l1l1l1 = l1*l1*l1;
    int grid_slice[6];
    double *xs_exp, *ys_exp, *zs_exp, *exp_corr;
    size_t data_size = init_nonorth_data(&xs_exp, &ys_exp, &zs_exp, &exp_corr,
                                         grid_slice, dh, dh_inv, mesh, topl, cutoff,
                                         ai, aj, ri, rj, cache);
    if (data_size == 0) {
            return 0;
    }
    cache += data_size;

    double *mat_xyz = cache;
    double *mat_ijk = mat_xyz + l1l1l1;
    cache = mat_ijk + l1l1l1;

    memset(mat_xyz, 0, l1l1l1*sizeof(double));
    memset(mat_ijk, 0, l1l1l1*sizeof(double));

    _nonorth_ints(mat_ijk, weights, topl, fac, xs_exp, ys_exp, zs_exp, exp_corr,
                  grid_slice, mesh, cache);

    dm_ijk_to_dm_xyz(mat_ijk, mat_xyz, dh, topl);

    cache = mat_xyz + l1l1l1;
    double *pout_x = out;
    double *pout_y = pout_x + dij;
    double *pout_z = pout_y + dij;
    _v1_xyz_to_v1(_vrho_loop_ip1_x, mat_xyz, pout_x, li, lj, ai, aj, ri, rj, cache);
    _v1_xyz_to_v1(_vrho_loop_ip1_y, mat_xyz, pout_y, li, lj, ai, aj, ri, rj, cache);
    _v1_xyz_to_v1(_vrho_loop_ip1_z, mat_xyz, pout_z, li, lj, ai, aj, ri, rj, cache);
    return 1;
}


static size_t _orth_ints_core_cache_size(int* mesh, double radius, double* dh, int comp)
{
    size_t size = 0;
    size_t nmx = get_max_num_grid_orth(dh, radius);
    int max_mesh = MAX(MAX(mesh[0], mesh[1]), mesh[2]);
    const int l = 0;
    int l1 = l + 1;
    if (comp == 3) {
        l1 += 1;
    }
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l1];

    size_t size_orth_components = l1 * nmx + nmx;
    size_t size_orth_ints = 0;
    if (nmx < max_mesh) {
        size_orth_ints = (l1 + l1l1) * nmx;
    } else {
        size_orth_ints = l1*mesh[2] + l1l1*mesh[0];
    }
    size += MAX(size_orth_components, size_orth_ints);
    size += l1 * (mesh[0] + mesh[1] + mesh[2]);
    size += l1l1 * l1;
    size += 3 * (ncart + l1);
    return size;
}


static size_t _nonorth_ints_core_cache_size(int* mesh, double radius, double* dh, double* dh_inv, int comp)
{
    size_t size = 0;
    //size_t nmx = get_max_num_grid_nonorth(dh_inv, radius);
    size_t nmx = get_max_num_grid_nonorth_tight(dh, dh_inv, radius);
    size_t nmx2 = nmx * nmx;
    const int l = 0;
    int l1 = l + 1;
    if (comp == 3) {
        l1 += 1;
    }
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l1];

    size += l1 * nmx * 3; // xs_exp, ys_exp, zs_exp
    size += nmx2 * 3; //exp_corr

    size_t tmp = nmx * 2;
    if (l > 0) {
        tmp += nmx;
    }
    size_t tmp1 = l1l1 * l1 * 2; // dm_xyz, dm_ijk
    tmp1 += nmx2 * nmx + l1 * nmx2 + l1l1 * nmx; // _nonorth_ints
    tmp1 = MAX(tmp1, 3 * (ncart + l1)); // dm_xyz_to_dm
    size += MAX(tmp, tmp1);
    return size;
}


static size_t _ints_core_cache_size(int* mesh, double radius, double* dh, double *dh_inv, int comp, bool orth)
{
    if (orth) {
        return _orth_ints_core_cache_size(mesh, radius, dh, comp);
    } else {
        return _nonorth_ints_core_cache_size(mesh, radius, dh, dh_inv, comp);
    }
}


void int_gauss_charge_v_rs(int (*eval_ints)(), double* out, double* v_rs, int comp,
                           int* atm, int* bas, int nbas, double* env,
                           int* mesh, int dimension, double* a, double* b, double max_radius, bool orth)
{
    double dh[9], dh_inv[9];
    get_grid_spacing(dh, dh_inv, a, b, mesh);

    size_t cache_size = _ints_core_cache_size(mesh, max_radius, dh, dh_inv, comp, orth);

#pragma omp parallel
{
    int ia, ib;
    double alpha, coeff, charge, rad, fac;
    double *r0;
    double *cache = (double*) malloc(sizeof(double) * cache_size);
    #pragma omp for schedule(static)
    for (ib = 0; ib < nbas; ib++) {
        ia = bas[ib*BAS_SLOTS+ATOM_OF];
        alpha = env[bas[ib*BAS_SLOTS+PTR_EXP]];
        coeff = env[bas[ib*BAS_SLOTS+PTR_COEFF]];
        charge = (double)atm[ia*ATM_SLOTS+CHARGE_OF];
        r0 = env + atm[ia*ATM_SLOTS+PTR_COORD];
        fac = -charge * coeff;
        rad = env[atm[ia*ATM_SLOTS+PTR_RADIUS]];
        if (rad > 1e-15) {
            (*eval_ints)(v_rs, out+ia*comp, comp, 0, 0, alpha, 0.0, r0, r0,
                         fac, rad, dimension, dh, dh_inv, mesh, cache);
        }
    }
    free(cache);
}
}
