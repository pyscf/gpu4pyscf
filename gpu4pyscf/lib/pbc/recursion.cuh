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

#pragma once

__forceinline__ __device__
void vrr(double *gx, double *Rpq, double ai, double aj, double rt,
         int li, int lj, int gout_id, int gout_stride, int nsp_per_block)
{
    int stride_j = li + 1;
    int g_size = (li + 1) * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    double aij = ai + aj;
    double rt_aa = rt / aij;
    double s0x, s1x, s2x;
    if (li > 0) {
        __syncthreads();
        double rt_ai = rt_aa * aj;
        double b10 = .5/ai * (1 - rt_ai);
        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
        for (int n = gout_id; n < 3; n += gout_stride) {
            double *_gx = gx + n * gx_len;
            double c0x = -rt_ai * Rpq[n*nsp_per_block];
            s0x = _gx[0];
            s1x = c0x * s0x;
            _gx[nsp_per_block] = s1x;
            for (int i = 1; i < li; ++i) {
                s2x = c0x * s1x + i * b10 * s0x;
                _gx[(i+1)*nsp_per_block] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
        }
    }

    if (lj > 0) {
        int li3 = (li+1)*3;
        double rt_ak  = rt_aa * ai;
        double b00 = .5 * rt_aa;
        double b01 = .5/aj  * (1 - rt_ak);
        for (int n = gout_id; n < li3+gout_id; n += gout_stride) {
            __syncthreads();
            int i = n / 3; //for i in range(li+1):
            int _ix = n % 3; // TODO: remove _ix for nroots > 2
            double *_gx = gx + (i + _ix * g_size) * nsp_per_block;
            double cpx = rt_ak * Rpq[_ix*nsp_per_block];
            //for i in range(li+1):
            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
            if (n < li3) {
                s0x = _gx[0];
                s1x = cpx * s0x;
                if (i > 0) {
                    s1x += i * b00 * _gx[-nsp_per_block];
                }
                _gx[stride_j*nsp_per_block] = s1x;
            }
            for (int j = 1; j < lj; ++j) {
                __syncthreads();
                if (n < li3) {
                    s2x = cpx*s1x + j*b01*s0x;
                    if (i > 0) {
                        s2x += i * b00 * _gx[(j*stride_j-1)*nsp_per_block];
                    }
                    _gx[(j*stride_j+stride_j)*nsp_per_block] = s2x;
                    s0x = s1x;
                    s1x = s2x;
                }
            }
        }
    }
    __syncthreads();
}
