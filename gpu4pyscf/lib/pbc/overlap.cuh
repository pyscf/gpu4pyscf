/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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

#define PI_POW_1_5       5.568327996831707845
#define REMOTE_THRESHOLD 50

__inline__ __device__
void vrr_hrr(double *gx, double *rjri, double ai, double aj, double cicj,
             int li, int lj, int gout_id, int gout_stride, int nsp_per_block)
{
    int stride_j = li + 1;
    int g_size = (li + 1) * (lj + 1);
    int gx_len = g_size * nsp_per_block;
    double aij = ai + aj;
    double aj_aij = aj / aij;
    if (gout_id == 0) {
        double theta = ai * aj_aij;
        double theta_rr = theta * rjri[3*nsp_per_block];
        gx[gx_len*2] = cicj / (aij*sqrt(aij)) * exp(-theta_rr);
    }
    int lij = li + lj;
    if (lij > 0) {
        __syncthreads();
        double s0x, s1x, s2x;
        double b = .5 / aij;
        for (int n = gout_id; n < 3; n += gout_stride) {
            double *_gx = gx + n * gx_len;
            double xjxi = rjri[n*nsp_per_block];
            double xpa = xjxi * aj_aij;
            s0x = _gx[0];
            s1x = xpa * s0x;
            _gx[nsp_per_block] = s1x;
            for (int i = 1; i < lij; ++i) {
                s2x = xpa * s1x + i * b * s0x;
                _gx[(i+1)*nsp_per_block] = s2x;
                s0x = s1x;
                s1x = s2x;
            }
            for (int j = 0; j < lj; ++j) {
                int ij = (lij-j) + j*stride_j;
                s1x = _gx[ij*nsp_per_block];
                for (--ij; ij >= j*stride_j; --ij) {
                    s0x = _gx[ij*nsp_per_block];
                    _gx[(ij+stride_j)*nsp_per_block] = s1x - xjxi * s0x;
                    s1x = s0x;
                }
            }
        }
    }
    __syncthreads();
}

