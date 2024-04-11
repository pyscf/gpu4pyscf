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

#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS 128

// (n,ncart,stride) -> (n,nsph,stride), count = n*stride
__global__
static void _cart2sph_ang2(double *cart, double *sph, int stride, int count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count){
        return;
    }
    int i = idx / stride;
    int j = idx % stride;
    int sph_offset = 5 * stride * i + j;
    int cart_offset = 6 * stride * i + j;
    double g0 = cart[cart_offset+0*stride];
    double g1 = cart[cart_offset+1*stride];
    double g2 = cart[cart_offset+2*stride];
    double g3 = cart[cart_offset+3*stride];
    double g4 = cart[cart_offset+4*stride];
    double g5 = cart[cart_offset+5*stride];

    sph[sph_offset+0*stride] = 1.092548430592079070 * g1;
    sph[sph_offset+1*stride] = 1.092548430592079070 * g4;
    sph[sph_offset+2*stride] = 0.630783130505040012 * g5 - 0.315391565252520002 * (g0 + g3);
    sph[sph_offset+3*stride] = 1.092548430592079070 * g2;
    sph[sph_offset+4*stride] = 0.546274215296039535 * (g0 - g3);
}

__global__
static void _cart2sph_ang3(double *cart, double *sph, int stride, int count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count){
        return;
    }
    int i = idx / stride;
    int j = idx % stride;
    int sph_offset = 7 * stride * i + j;
    int cart_offset = 10 * stride * i + j;
    double g0 = cart[cart_offset+0*stride];
    double g1 = cart[cart_offset+1*stride];
    double g2 = cart[cart_offset+2*stride];
    double g3 = cart[cart_offset+3*stride];
    double g4 = cart[cart_offset+4*stride];
    double g5 = cart[cart_offset+5*stride];
    double g6 = cart[cart_offset+6*stride];
    double g7 = cart[cart_offset+7*stride];
    double g8 = cart[cart_offset+8*stride];
    double g9 = cart[cart_offset+9*stride];

    sph[sph_offset+0*stride] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
    sph[sph_offset+1*stride] = 2.890611442640554055 * g4;
    sph[sph_offset+2*stride] = 1.828183197857862944 * g8 - 0.457045799464465739 * (g1 + g6);
    sph[sph_offset+3*stride] = 0.746352665180230782 * g9 - 1.119528997770346170 * (g2 + g7);
    sph[sph_offset+4*stride] = 1.828183197857862944 * g5 - 0.457045799464465739 * (g0 + g3);
    sph[sph_offset+5*stride] = 1.445305721320277020 * (g2 - g7);
    sph[sph_offset+6*stride] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;
}

__global__
static void _cart2sph_ang4(double *cart, double *sph, int stride, int count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count){
        return;
    }
    int i = idx / stride;
    int j = idx % stride;
    int sph_offset = 9 * stride * i + j;
    int cart_offset = 15 * stride * i + j;
    double g0 = cart[cart_offset+0*stride];
    double g1 = cart[cart_offset+1*stride];
    double g2 = cart[cart_offset+2*stride];
    double g3 = cart[cart_offset+3*stride];
    double g4 = cart[cart_offset+4*stride];
    double g5 = cart[cart_offset+5*stride];
    double g6 = cart[cart_offset+6*stride];
    double g7 = cart[cart_offset+7*stride];
    double g8 = cart[cart_offset+8*stride];
    double g9 = cart[cart_offset+9*stride];
    double g10 = cart[cart_offset+10*stride];
    double g11 = cart[cart_offset+11*stride];
    double g12 = cart[cart_offset+12*stride];
    double g13 = cart[cart_offset+13*stride];
    double g14 = cart[cart_offset+14*stride];

    sph[sph_offset+0*stride] = 2.503342941796704538 * (g1 - g6);
    sph[sph_offset+1*stride] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
    sph[sph_offset+2*stride] = 5.677048174545360108 * g8 - 0.946174695757560014 * (g1 + g6);
    sph[sph_offset+3*stride] = 2.676186174229156671 * g13- 2.007139630671867500 * (g4 + g11);
    sph[sph_offset+4*stride] = 0.317356640745612911 * (g0 + g10) + 0.634713281491225822 * g3 - 2.538853125964903290 * (g5 + g12) + 0.846284375321634430 * g14;
    sph[sph_offset+5*stride] = 2.676186174229156671 * g9 - 2.007139630671867500 * (g2 + g7);
    sph[sph_offset+6*stride] = 2.838524087272680054 * (g5 - g12) + 0.473087347878780009 * (g10 - g0);
    sph[sph_offset+7*stride] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
    sph[sph_offset+8*stride] = 0.625835735449176134 * (g0 + g10) - 3.755014412695056800 * g3;
}

extern "C" {
__host__
int cart2sph(cudaStream_t stream, double *cart_gto, double *sph_gto, int stride, int count, int ang)
{
    dim3 threads(THREADS);
    dim3 blocks((count + THREADS - 1)/THREADS);
    switch (ang) {
        case 0: break;
        case 1: break;
        case 2: _cart2sph_ang2 <<<blocks, threads, 0, stream>>> (cart_gto, sph_gto, stride, count); break;
        case 3: _cart2sph_ang3 <<<blocks, threads, 0, stream>>> (cart_gto, sph_gto, stride, count); break;
        case 4: _cart2sph_ang4 <<<blocks, threads, 0, stream>>> (cart_gto, sph_gto, stride, count); break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
