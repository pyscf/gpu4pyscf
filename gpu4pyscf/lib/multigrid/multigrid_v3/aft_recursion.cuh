/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
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

#define NGV_PER_BLOCK   16

__forceinline__ __device__
void vrr_hrr(double *gx, double *swap, int addrR, int li, int lj,
             int stride_j, double a2, double xjxi, double aj_aij,
             double xi, double kx, double theta_rr)
{
    constexpr int stride_i = NGV_PER_BLOCK * 6;
    int addrI = addrR + NGV_PER_BLOCK;
    int lij = li + lj;
    double xpa = xjxi * aj_aij;
    double xij = xpa + xi;
    double kR = kx * xij;
    double s0xR, s1xR, s2xR;
    double s0xI, s1xI, s2xI;
    sincos(-kR, &s0xI, &s0xR);
    double Kab = exp(-theta_rr);
    s0xR *= Kab;
    s0xI *= Kab;
    swap[addrR] = s0xR;
    swap[addrI] = s0xI;
    gx[addrR] += s0xR;
    gx[addrI] += s0xI;
    if (lij > 0) {
        double RpaR = xpa;
        double RpaI = -a2 * kx;
        s1xR = RpaR * s0xR - RpaI * s0xI;
        s1xI = RpaR * s0xI + RpaI * s0xR;
        swap[addrR+stride_i] = s1xR;
        swap[addrI+stride_i] = s1xI;
        if (0 < li) {
            gx[addrR+stride_i] += s1xR;
            gx[addrI+stride_i] += s1xI;
        }
        for (int i = 2; i <= lij; i++) {
            double ia2 = (i-1) * a2;
            s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
            s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
            swap[addrR+i*stride_i] = s2xR;
            swap[addrI+i*stride_i] = s2xI;
            if (i <= li) {
                gx[addrR+i*stride_i] += s2xR;
                gx[addrI+i*stride_i] += s2xI;
            }
            s0xR = s1xR;
            s0xI = s1xI;
            s1xR = s2xR;
            s1xI = s2xI;
        }
    }
    for (int j = 1; j <= lj; ++j) {
        int i = lij - j;
        s1xR = swap[addrR+(i+1)*stride_i];
        s1xI = swap[addrI+(i+1)*stride_i];
        for (; i >= 0; --i) {
            s0xR = swap[addrR+i*stride_i];
            s0xI = swap[addrI+i*stride_i];
            s2xR = s1xR - xjxi * s0xR;
            s2xI = s1xI - xjxi * s0xI;
            swap[addrR+i*stride_i] = s2xR;
            swap[addrI+i*stride_i] = s2xI;
            if (i <= li) {
                int ij = i * stride_i + j * stride_j;
                gx[addrR+ij] += s2xR;
                gx[addrI+ij] += s2xI;
            }
            s1xR = s0xR;
            s1xI = s0xI;
        }
    }
}

template <int LI, int LJ> __forceinline__ __device__
void vrr_hrr(double *gx, int addrR, int stride_j, double a2, double xjxi,
             double RpaR, double RpaI, double g00R, double g00I)
{
    constexpr int stride_i = NGV_PER_BLOCK * 6;
    int addrI = addrR + NGV_PER_BLOCK;
    double swapR[LI+LJ+1];
    double swapI[LI+LJ+1];
    double s0xR, s1xR, s2xR;
    double s0xI, s1xI, s2xI;
    s0xR = g00R;
    s0xI = g00I;
    swapR[0] = s0xR;
    swapI[0] = s0xI;
    gx[addrR] += s0xR;
    gx[addrI] += s0xI;
    constexpr int lij = LI + LJ;
    if (lij > 0) {
        s1xR = RpaR * s0xR - RpaI * s0xI;
        s1xI = RpaR * s0xI + RpaI * s0xR;
        swapR[1] = s1xR;
        swapI[1] = s1xI;
        if (0 < LI) {
            gx[addrR+stride_i] += s1xR;
            gx[addrI+stride_i] += s1xI;
        }
#pragma unroll
        for (int i = 2; i <= lij; i++) {
            double ia2 = (i-1) * a2;
            s2xR = ia2 * s0xR + RpaR * s1xR - RpaI * s1xI;
            s2xI = ia2 * s0xI + RpaR * s1xI + RpaI * s1xR;
            swapR[i] = s2xR;
            swapI[i] = s2xI;
            if (i <= LI) {
                gx[addrR+i*stride_i] += s2xR;
                gx[addrI+i*stride_i] += s2xI;
            }
            s0xR = s1xR;
            s0xI = s1xI;
            s1xR = s2xR;
            s1xI = s2xI;
        }
    }
#pragma unroll
    for (int j = 1; j <= LJ; ++j) {
        int i = lij - j;
        s1xR = swapR[i+1];
        s1xI = swapI[i+1];
#pragma unroll
        for (; i >= 0; --i) {
            s0xR = swapR[i];
            s0xI = swapI[i];
            s2xR = s1xR - xjxi * s0xR;
            s2xI = s1xI - xjxi * s0xI;
            swapR[i] = s2xR;
            swapI[i] = s2xI;
            if (i <= LI) {
                int ij = i * stride_i + j * stride_j;
                gx[addrR+ij] += s2xR;
                gx[addrI+ij] += s2xI;
            }
            s1xR = s0xR;
            s1xI = s0xI;
        }
    }
}

__forceinline__ __device__
void dI_gx(double *gx, int addr, int stride_i, int li,
           double ai2, double &outR, double &outI)
{
    outR = ai2 * gx[addr+stride_i];
    outI = ai2 * gx[addr+stride_i+NGV_PER_BLOCK];
    if (li > 0) {
        outR += li * gx[addr-stride_i];
        outI += li * gx[addr-stride_i+NGV_PER_BLOCK];
    }
}

__forceinline__ __device__
void dIdJ_gx(double *gx, int addr, int stride_i, int stride_j, int li, int lj,
             double ai2, double aj2, double &outR, double &outI)
{
    outR = ai2 * gx[addr+stride_i+stride_j];
    outI = ai2 * gx[addr+stride_i+stride_j+NGV_PER_BLOCK];
    if (li > 0) {
        outR += li * gx[addr-stride_i+stride_j];
        outI += li * gx[addr-stride_i+stride_j+NGV_PER_BLOCK];
    }
    outR *= aj2;
    outI *= aj2;
    if (lj > 0) {
        double f1R = ai2 * gx[addr+stride_i-stride_j];
        double f1I = ai2 * gx[addr+stride_i-stride_j+NGV_PER_BLOCK];
        if (li > 0) {
            f1R += li * gx[addr-stride_i-stride_j];
            f1I += li * gx[addr-stride_i-stride_j+NGV_PER_BLOCK];
        }
        outR += lj * f1R;
        outI += lj * f1I;
    }
}

__forceinline__ __device__
void dIdJ_gx(double *gx, int addr, int stride_i, int li,
             double ai2, double kx, double &outR, double &outI)
{
    int li2 = li * 2 + 1;
    outR = li2 * gx[addr];
    outI = li2 * gx[addr+NGV_PER_BLOCK];
    outR += ai2 * gx[addr+stride_i*2];
    outI += ai2 * gx[addr+stride_i*2+NGV_PER_BLOCK];
    if (li > 1) {
        int lili = li * (li-1);
        outR += lili * gx[addr-stride_i*2];
        outI += lili * gx[addr-stride_i*2+NGV_PER_BLOCK];
    }

    // (d i|d j) + (d2 i| j) + -iG*(d i|j) = 0
    double f1R = ai2 * gx[addr+stride_i];
    double f1I = ai2 * gx[addr+stride_i+NGV_PER_BLOCK];
    if (li > 0) {
        f1R += li * gx[addr-stride_i];
        f1I += li * gx[addr-stride_i+NGV_PER_BLOCK];
    }
    outR = f1I * -kx - outR;
    outI = f1R *  kx - outI;
}

__forceinline__ __device__
void d2IdJ_gx(double *gx, int addr, int stride_i, int stride_j, int li, int lj,
              double ai2, double aj2, double &outR, double &outI)
{
    int li2 = li * 2 + 1;
    outR = li2 * gx[addr+stride_j];
    outI = li2 * gx[addr+stride_j+NGV_PER_BLOCK];
    outR += ai2 * gx[addr+stride_i*2+stride_j];
    outI += ai2 * gx[addr+stride_i*2+stride_j+NGV_PER_BLOCK];
    if (li > 1) {
        int lili = li * (li-1);
        outR += lili * gx[addr-stride_i*2+stride_j];
        outI += lili * gx[addr-stride_i*2+stride_j+NGV_PER_BLOCK];
    }
    outR *= aj2;
    outI *= aj2;
    if (lj > 0) {
        int li2 = li * 2 + 1;
        double f1R = li2 * gx[addr-stride_j];
        double f1I = li2 * gx[addr-stride_j+NGV_PER_BLOCK];
        f1R += ai2 * gx[addr+stride_i*2-stride_j];
        f1I += ai2 * gx[addr+stride_i*2-stride_j+NGV_PER_BLOCK];
        if (li > 1) {
            int lili = li * (li-1);
            f1R += lili * gx[addr-stride_i*2-stride_j];
            f1I += lili * gx[addr-stride_i*2-stride_j+NGV_PER_BLOCK];
        }
        outR += lj * f1R;
        outI += lj * f1I;
    }
}
