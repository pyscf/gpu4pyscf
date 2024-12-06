/* Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

#include "gvhf-rys/rys_roots.cuh"

#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096

template <int NROOTS> __device__
static void GINTrys_root(double x, double *rw)
{
    constexpr int off = NROOTS * (NROOTS - 1) / 2;
    const double t = sqrt(PIE4/x);
    
    if (x<3.0e-7){
        for (int rt_id = 0; rt_id < NROOTS; ++rt_id) {
            const double r = ROOT_SMALLX_R0[off+rt_id] + ROOT_SMALLX_R1[off+rt_id] * x;
            const double w = ROOT_SMALLX_W0[off+rt_id] + ROOT_SMALLX_W1[off+rt_id] * x;
            rw[rt_id] = r / (1 - r);
            rw[rt_id+NROOTS] = w;
        }
        return;
    }

    if (x>35+NROOTS*5){
        for (int rt_id = 0; rt_id < NROOTS; ++rt_id) {
            const double r = ROOT_LARGEX_R_DATA[off+rt_id] / x;
            const double w = ROOT_LARGEX_W_DATA[off+rt_id] * t;
            rw[rt_id] = r / (1 - r);
            rw[rt_id+NROOTS] = w;
        }
        return;
    }

    for (int rt_id = 0; rt_id < NROOTS; ++rt_id) {
        const int it = (int)(x * .4);
        double *datax = ROOT_RW_DATA + DEGREE1*INTERVALS * NROOTS*(NROOTS-1);
        const double u = (x - it * 2.5) * 0.8 - 1.;
        const double u2 = u * 2.;
        double *c = datax + (2*rt_id) * DEGREE1 * INTERVALS;
        //for i in range(2, degree + 1):
        //    c0, c1 = c[degree-i] - c1, c0 + c1*u2
        double c0 = c[it + DEGREE   *INTERVALS];
        double c1 = c[it +(DEGREE-1)*INTERVALS];
        double c2, c3;

        double r, w;
#pragma unroll
        for (int n = DEGREE-2; n > 0; n-=2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        if (DEGREE % 2 == 0) {
            c2 = c[it] - c1;
            c3 = c0 + c1*u2;
            r = c2 + c3*u;
        } else {
            r = c0 + c1*u;
        }

        // For weights
        c = datax + (2*rt_id+1) * DEGREE1 * INTERVALS;
        c0 = c[it + DEGREE   *INTERVALS];
        c1 = c[it +(DEGREE-1)*INTERVALS];
        
#pragma unroll
        for (int n = DEGREE-2; n > 0; n-=2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        if (DEGREE % 2 == 0) {
            c2 = c[it] - c1;
            c3 = c0 + c1*u2;
            w = c2 + c3*u;
        } else {
            w = c0 + c1*u;
        }

        rw[rt_id] = r / (1 - r);
        rw[rt_id+NROOTS] = w;
    }
}

template<> __device__
inline void GINTrys_root<1>(double x, double *rw)
{
    double tt = sqrt(x);
    double fmt0 = SQRTPIE4 / tt * erf(tt);
    rw[1] = fmt0;
    double e = exp(-x);
    double b = .5 / x;
    double fmt1 = b * (fmt0 - e);
    rw[0] = fmt1 / (fmt0 - fmt1);
    return;
}

template <int NROOTS> __device__
inline void GINTscale_u(double *u, double theta)
{
# pragma unroll
    for(int i = 0; i < NROOTS; i++){
        u[i] /= u[i] + 1 - u[i] * theta;
    }
}
