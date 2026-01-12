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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

int host_Rt2_idx_offsets[81];
__device__ int Rt2_idx_offsets[81];
__device__ uint16_t Rt2_ij_kl[245025];
__device__ uint16_t Rt2_kl_ij[245025];
__device__ int8_t Rt2_efg_phase[165];
__device__ int8_t Rt_tuv_fac[4828];
__device__ uint16_t Rt_idx[4828];

void l3_address_lookup(int *lut, int l)
{
    int l1 = l + 1;
    int i = 0;
    for (int x = 0; x <= l; x++) {
        for (int y = 0; y <= l - x; y++) {
            for (int z = 0; z <= l - x - y; z++, i++) {
                lut[x * l1 * l1 + y * l1 + z] = i;
            }
        }
    }
}

int Rt_iter_indices(uint16_t *Rt_idx, int8_t *fac, int *lut, int l)
{
    int l1 = l + 1;
    int n = 0;
    Rt_idx[n] = 0;
    fac[n] = 0;
    n++;
    for (int v = 1; v <= l; v++, n++) {
        Rt_idx[n] = lut[v-1];
        fac[n] = v;
    }

    for (int u = 0; u <= l; u++) {
        if (u == 0) {
            for (int v = 0; v <= l - u; v++, n++) {
                Rt_idx[n] = 0;
                fac[n] = u;
            }
        } else {
            for (int v = 0; v <= l - u; v++, n++) {
                Rt_idx[n] = lut[(u-1)*l1+v];
                fac[n] = u;
            }
        }
    }

    for (int t = 0; t <= l; t++) {
        if (t == 0) {
            for (int u = 0; u <= l - t; u++) {
                for (int v = 0; v <= l - t - u; v++, n++) {
                    Rt_idx[n] = 0;
                    fac[n] = t;
                }
            }
        } else {
            for (int u = 0; u <= l - t; u++) {
                for (int v = 0; v <= l - t - u; v++, n++) {
                    Rt_idx[n] = lut[((t-1)*l1+u)*l1+v];
                    fac[n] = t;
                }
            }
        }
    }
    return n;
}

int Rt2_iter_indices(uint16_t *Rt2_ij_kl, uint16_t *Rt2_kl_ij, int *lut, int lij, int lkl)
{
    int n = 0;
    int l1 = lij+lkl + 1;
    for (int e = 0; e <= lkl; e++) {
        for (int f = 0; f <= lkl - e; f++) {
            for (int g = 0; g <= lkl - e - f; g++) {
                for (int t = 0; t <= lij; t++) {
                    for (int u = 0; u <= lij - t; u++) {
                        for (int v = 0; v <= lij - t - u; v++, n++) {
                            Rt2_kl_ij[n] = lut[((t+e)*l1+u+f)*l1+v+g];
                        }
                    }
                }
            }
        }
    }
    n = 0;
    for (int t = 0; t <= lij; t++) {
        for (int u = 0; u <= lij - t; u++) {
            for (int v = 0; v <= lij - t - u; v++) {
                for (int e = 0; e <= lkl; e++) {
                    for (int f = 0; f <= lkl - e; f++) {
                        for (int g = 0; g <= lkl - e - f; g++, n++) {
                            Rt2_ij_kl[n] = lut[((t+e)*l1+u+f)*l1+v+g];
                        }
                    }
                }
            }
        }
    }
    return n;
}

void initialize_Rt2_indices()
{
    constexpr int lmax2 = LMAX * 2;
    constexpr int lmax4 = LMAX * 4;
    int lut_buf_size = 0;
    for (int l = 0; l <= lmax4; l++) {
        int l1 = l + 1;
        lut_buf_size += l1 * l1 * l1;
    }
    int n = 0;
    int *lut[lmax4+1];
    int *lut_buf = (int *)malloc(sizeof(int) * lut_buf_size);
    for (int l = 0; l <= lmax4; l++) {
        int l1 = l + 1;
        lut[l] = lut_buf + n;
        n += l1 * l1 * l1;
        l3_address_lookup(lut[l], l);
    }

    int nf2 = (lmax2+1)*(lmax2+2)*(lmax2+3)*(lmax2+4)/24;
    uint16_t *_Rt2_ij_kl = (uint16_t *)malloc(sizeof(uint16_t) * nf2*nf2);
    uint16_t *_Rt2_kl_ij = (uint16_t *)malloc(sizeof(uint16_t) * nf2*nf2);
    n = 0;
    for (int lij = 0; lij <= lmax2; lij++) {
        for (int lkl = 0; lkl <= lmax2; lkl++) {
            host_Rt2_idx_offsets[lij*(lmax2+1)+lkl] = n;
            n += Rt2_iter_indices(_Rt2_ij_kl+n, _Rt2_kl_ij+n, lut[lij+lkl], lij, lkl);
        }
    }
    cudaMemcpy(Rt2_idx_offsets, host_Rt2_idx_offsets, sizeof(int)*(lmax2+1)*(lmax2+1), cudaMemcpyHostToDevice);
    cudaMemcpy(Rt2_ij_kl, _Rt2_ij_kl, sizeof(uint16_t)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(Rt2_kl_ij, _Rt2_kl_ij, sizeof(uint16_t)*n, cudaMemcpyHostToDevice);

    int8_t *_Rt2_efg_phase = (int8_t *)malloc(sizeof(int8_t) * nf2);
    n = 0;
    for (int l = 0; l <= lmax2; l++) {
        for (int e = 0; e <= l; e++) {
            for (int f = 0; f <= l - e; f++) {
                for (int g = 0; g <= l - e - f; g++, n++) {
                    _Rt2_efg_phase[n] = ((e + f + g) % 2 == 0) ? 1 : -1;
                }
            }
        }
    }
    cudaMemcpy(Rt2_efg_phase, _Rt2_efg_phase, sizeof(int8_t)*n, cudaMemcpyHostToDevice);

    int nf4 = (lmax4+1)*(lmax4+2)*(lmax4+3)*(lmax4+4)/24;
    // offsets = l*(l+1)*(l+2)*(l+3)//24 - l
    uint16_t *_Rt_idx = (uint16_t *)malloc(sizeof(uint16_t) * (nf4-lmax4-1));
    int8_t *_Rt_tuv_fac = (int8_t *)malloc(sizeof(int8_t) * (nf4-lmax4-1));
    n = 0;
    for (int l = 0; l < lmax4; l++) {
        n += Rt_iter_indices(_Rt_idx+n, _Rt_tuv_fac+n, lut[l], l);
    }
    cudaMemcpy(Rt_idx, _Rt_idx, sizeof(uint16_t)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(Rt_tuv_fac, _Rt_tuv_fac, sizeof(int8_t)*n, cudaMemcpyHostToDevice);

    free(_Rt_idx);
    free(_Rt_tuv_fac);
    free(_Rt2_ij_kl);
    free(_Rt2_kl_ij);
    free(_Rt2_efg_phase);
    free(lut_buf);
}
