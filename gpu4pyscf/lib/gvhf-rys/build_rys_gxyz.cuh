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


#define BUILD_4C_GXYZ(lj, ll, active) \
        __syncthreads(); \
        int gx_len = nsq_per_block * g_size; \
        if (gout_id == 0) { \
            gx[gx_len*2] = rw[(irys*2+1)*nsq_per_block]; \
        } \
        double rt = rw[irys*2*nsq_per_block]; \
        double aij = aij_cache[0]; \
        double rt_aa = rt / (aij + akl); \
        double s0x, s1x, s2x; \
        if (lij > 0) { \
            double aj_aij = aij_cache[1]; \
            double rt_aij = rt_aa * akl; \
            double b10 = .5/aij * (1 - rt_aij); \
            __syncthreads(); \
            for (int n = gout_id; n < 3; n += gout_stride) { \
                double *_gx = gx + n * gx_len; \
                double Rpa = rjri[n] * aj_aij; \
                double c0x = Rpa - rt_aij * Rpq[n*nsq_per_block]; \
                s0x = _gx[0]; \
                s1x = c0x * s0x; \
                _gx[nsq_per_block] = s1x; \
                for (int i = 1; i < lij; ++i) { \
                    s2x = c0x * s1x + i * b10 * s0x; \
                    _gx[(i+1)*nsq_per_block] = s2x; \
                    s0x = s1x; \
                    s1x = s2x; \
                } \
            } \
        } \
        if (lkl > 0) { \
            double rt_akl = rt_aa * aij; \
            double b00 = .5 * rt_aa; \
            double b01 = .5/akl * (1 - rt_akl); \
            int lij3 = (lij+1)*3; \
            for (int n = gout_id; n < lij3+gout_id; n += gout_stride) { \
                __syncthreads(); \
                int i = n / 3; \
                int _ix = n - i * 3; \
                double *_gx = gx + (i + _ix * g_size) * nsq_per_block; \
                double Rqc = rlrk[_ix*nsq_per_block] * al_akl; \
                double cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block]; \
                if (n < lij3) { \
                    s0x = _gx[0]; \
                    s1x = cpx * s0x; \
                    if (i > 0) { \
                        s1x += i * b00 * _gx[-nsq_per_block]; \
                    } \
                    _gx[stride_k*nsq_per_block] = s1x; \
                } \
                for (int k = 1; k < lkl; ++k) { \
                    __syncthreads(); \
                    if (n < lij3) { \
                        s2x = cpx*s1x + k*b01*s0x; \
                        if (i > 0) { \
                            s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block]; \
                        } \
                        _gx[(k*stride_k+stride_k)*nsq_per_block] = s2x; \
                        s0x = s1x; \
                        s1x = s2x; \
                    } \
                } \
            } \
        } \
        if (lj > 0) { \
            __syncthreads(); \
            if (active) { \
                int lkl3 = (lkl+1)*3; \
                for (int m = gout_id; m < lkl3; m += gout_stride) { \
                    int k = m / 3; \
                    int _ix = m - k * 3; \
                    double xjxi = rjri[_ix]; \
                    double *_gx = gx + (_ix*g_size + k*stride_k) * nsq_per_block; \
                    for (int j = 0; j < lj; ++j) { \
                        int ij = (lij-j) + j*stride_j; \
                        s1x = _gx[ij*nsq_per_block]; \
                        for (--ij; ij >= j*stride_j; --ij) { \
                            s0x = _gx[ij*nsq_per_block]; \
                            _gx[(ij+stride_j)*nsq_per_block] = s1x - xjxi * s0x; \
                            s1x = s0x; \
                        } \
                    } \
                } \
            } \
        } \
        if (ll > 0) { \
            __syncthreads(); \
            if (active) { \
                for (int n = gout_id; n < stride_k*3; n += gout_stride) { \
                    int i = n / 3; \
                    int _ix = n - i * 3; \
                    double xlxk = rlrk[_ix*nsq_per_block]; \
                    double *_gx = gx + (_ix*g_size + i) * nsq_per_block; \
                    for (int l = 0; l < ll; ++l) { \
                        int kl = (lkl-l)*stride_k + l*stride_l; \
                        s1x = _gx[kl*nsq_per_block]; \
                        for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) { \
                            s0x = _gx[kl*nsq_per_block]; \
                            _gx[(kl+stride_l)*nsq_per_block] = s1x - xlxk * s0x; \
                            s1x = s0x; \
                        } \
                    } \
                } \
            } \
        } \
        __syncthreads()

#define BUILD_3C_GXYZ(lj, lk, rjri_stride, active) \
        __syncthreads(); \
        int nst = nst_per_block; \
        int gx_len = nst * g_size; \
        if (gout_id == 0) { \
            gx[gx_len*2] = rw[(irys*2+1)*nst]; \
        } \
        double rt = rw[irys*2*nst]; \
        double rt_aa = rt / (aij + ak); \
        double s0x, s1x, s2x; \
        if (lij > 0) { \
            double rt_aij = rt_aa * ak; \
            double b10 = .5/aij * (1 - rt_aij); \
            __syncthreads(); \
            for (int n = gout_id; n < 3; n += gout_stride) { \
                double *_gx = gx + n * gx_len; \
                double Rpa = rjri[n*rjri_stride] * aj_aij; \
                double c0x = Rpa - rt_aij * Rpq[n*nst]; \
                s0x = _gx[0]; \
                s1x = c0x * s0x; \
                _gx[nst] = s1x; \
                for (int i = 1; i < lij; ++i) { \
                    s2x = c0x * s1x + i * b10 * s0x; \
                    _gx[(i+1)*nst] = s2x; \
                    s0x = s1x; \
                    s1x = s2x; \
                } \
            } \
        } \
        if (lk > 0) { \
            double rt_ak  = rt_aa * aij; \
            double b00 = .5 * rt_aa; \
            double b01 = .5/ak  * (1 - rt_ak ); \
            int lij3 = (lij+1)*3; \
            for (int n = gout_id; n < lij3+gout_id; n += gout_stride) { \
                __syncthreads(); \
                int i = n / 3; \
                int _ix = n - i * 3; \
                double *_gx = gx + (i + _ix * g_size) * nst; \
                double cpx = rt_ak * Rpq[_ix*nst]; \
                if (n < lij3) { \
                    s0x = _gx[0]; \
                    s1x = cpx * s0x; \
                    if (i > 0) { \
                        s1x += i * b00 * _gx[-nst]; \
                    } \
                    _gx[stride_k*nst] = s1x; \
                } \
                for (int k = 1; k < lk; ++k) { \
                    __syncthreads(); \
                    if (n < lij3) { \
                        s2x = cpx*s1x + k*b01*s0x; \
                        if (i > 0) { \
                            s2x += i * b00 * _gx[(k*stride_k-1)*nst]; \
                        } \
                        _gx[(k*stride_k+stride_k)*nst] = s2x; \
                        s0x = s1x; \
                        s1x = s2x; \
                    } \
                } \
            } \
        } \
        if (lj > 0) { \
            __syncthreads(); \
            if (active) { \
                int lk3 = (lk+1)*3; \
                for (int m = gout_id; m < lk3; m += gout_stride) { \
                    int k = m / 3; \
                    int _ix = m - k * 3; \
                    double xjxi = rjri[_ix*rjri_stride]; \
                    double *_gx = gx + (_ix*g_size + k*stride_k) * nst; \
                    for (int j = 0; j < lj; ++j) { \
                        int ij = (lij-j) + j*stride_j; \
                        s1x = _gx[ij*nst]; \
                        for (--ij; ij >= j*stride_j; --ij) { \
                            s0x = _gx[ij*nst]; \
                            _gx[(ij+stride_j)*nst] = s1x - xjxi * s0x; \
                            s1x = s0x; \
                        } \
                    } \
                } \
            } \
        } \
        __syncthreads()
