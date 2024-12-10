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

__global__
static void _cart2sph_ang5(double *cart, double *sph, int stride, int count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count){
        return;
    }
    int i = idx / stride;
    int j = idx % stride;
    int sph_offset = 11 * stride * i + j;
    int cart_offset = 21 * stride * i + j;
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
    double g15 = cart[cart_offset+15*stride];
    double g16 = cart[cart_offset+16*stride];
    double g17 = cart[cart_offset+17*stride];
    double g18 = cart[cart_offset+18*stride];
    double g19 = cart[cart_offset+19*stride];
    double g20 = cart[cart_offset+20*stride];
    sph[sph_offset+0*stride] = 3.2819102842008507 * (g1 - 2.0 * g6) + 0.6563820568401701 * g15;
    sph[sph_offset+1*stride] = 8.3026492595241645 * (g4 - g11);
    sph[sph_offset+2*stride] = -1.4677148983057511 * g1 + 11.7417191864460086 * g8 + 0.4892382994352504 * (g15 - 2.0 * g6) + -3.9139063954820030 * g17;
    sph[sph_offset+3*stride] = -4.7935367849733241 * (g4 + g11) + 9.5870735699466483 * g13;
    sph[sph_offset+4*stride] = 0.4529466511956969 * (g1 + g15 + 2.0 * g6) + -5.4353598143483630 * (g8 + g17) + 3.6235732095655755 * g19;
    sph[sph_offset+5*stride] = 1.7542548368013540 * (g2 + g16) + 3.5085096736027079 * g7 + -4.6780128981369442 * (g9 + g18) + 0.9356025796273888 * g20;
    sph[sph_offset+6*stride] = 0.4529466511956969 * (g0 + g10 + 2.0 * g3) + -5.4353598143483630 * (g5 + g12) + 3.6235732095655755 * g14;
    sph[sph_offset+7*stride] = -2.3967683924866621 * (g2 - g16) + 4.7935367849733241 * (g9 - g18);
    sph[sph_offset+8*stride] = -0.4892382994352504 * (g0 - 2.0 * g3) + 3.9139063954820030 * g5 + 1.4677148983057511 * g10 + -11.7417191864460086 * g12;
    sph[sph_offset+9*stride] = 2.0756623148810411 * (g2 + g16) + -12.4539738892862477 * g7;
    sph[sph_offset+10*stride] = 0.6563820568401701 * g0 + 3.2819102842008507 * (g10 - 2.0*g3);
}

__global__
static void _cart2sph_ang6(double *cart, double *sph, int stride, int count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count){
        return;
    }
    int i = idx / stride;
    int j = idx % stride;
    int sph_offset = 13 * stride * i + j;
    int cart_offset = 28 * stride * i + j;
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
    double g15 = cart[cart_offset+15*stride];
    double g16 = cart[cart_offset+16*stride];
    double g17 = cart[cart_offset+17*stride];
    double g18 = cart[cart_offset+18*stride];
    double g19 = cart[cart_offset+19*stride];
    double g20 = cart[cart_offset+20*stride];
    double g21 = cart[cart_offset+21*stride];
    double g22 = cart[cart_offset+22*stride];
    double g23 = cart[cart_offset+23*stride];
    double g24 = cart[cart_offset+24*stride];
    double g25 = cart[cart_offset+25*stride];
    double g26 = cart[cart_offset+26*stride];
    double g27 = cart[cart_offset+27*stride];
    sph[sph_offset+0*stride] = 4.0991046311514863 * (g1 + g15) + -13.6636821038382887 * g6;
    sph[sph_offset+1*stride] = 11.8330958111587634 * g4 + -23.6661916223175268 * g11 + 2.3666191622317525 * g22;
    sph[sph_offset+2*stride] = -2.0182596029148963 * (g1 - g15) + 20.1825960291489679 * (g8 - g17);
    sph[sph_offset+3*stride] = -8.2908473356343109 * g4 + -5.5272315570895412 * g11 + 22.1089262283581647 * g13 + 2.7636157785447706 * g22 + -7.3696420761193888 * g24;
    sph[sph_offset+4*stride] = 0.9212052595149236 * (g1 + g15 + 2.0 * g6) + -14.7392841522387776 * (g8 + g17 - g19);
    sph[sph_offset+5*stride] = 2.9131068125936568 * (g4 + g22) + 5.8262136251873136 * g11 + -11.6524272503746271 * (g13 + g24) + 4.6609709001498505 * g26;
    sph[sph_offset+6*stride] = -0.3178460113381421 * (g0 + g21 + 3.0*g3 + 3.0*g10) + 5.7212282040865583 * (g5 + g23) + 11.4424564081731166 * g12 + -7.6283042721154111 * (g14 + g25) + 1.0171072362820548 * g27;
    sph[sph_offset+7*stride] = 2.9131068125936568 * (g2 + g16) + 5.8262136251873136 * g7 + -11.6524272503746271 * (g9 + g18) + 4.6609709001498505 * g20;
    sph[sph_offset+8*stride] = 0.4606026297574618 * (g0 - g10) + 0.4606026297574618 * (g3 - g21) + -7.3696420761193888 * (g5 - g14 - g23 + g25);
    sph[sph_offset+9*stride] = -2.7636157785447706 * (g2 - 2.0 * g7) + 7.3696420761193888 * g9 + 8.2908473356343109 * g16 + -22.1089262283581647 * g18;
    sph[sph_offset+10*stride] = -0.5045649007287241 * (g0 + g21) + 2.5228245036436201 * (g3 + g10) + 5.0456490072872420 * (g5 + g23) + -30.2738940437234518 * g12;
    sph[sph_offset+11*stride] = 2.3666191622317525 * g2 + 11.8330958111587634 * (g16 - 2.0 * g7);
    sph[sph_offset+12*stride] = 0.6831841051919144 * (g0 - g21) + -10.2477615778787161 * (g3 - g10);
}

__global__
static void _cart2sph_ang7(double *cart, double *sph, int stride, int count){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count){
        return;
    }
    int i = idx / stride;
    int j = idx % stride;
    int sph_offset = 15 * stride * i + j;
    int cart_offset =36 * stride * i + j;
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
    double g15 = cart[cart_offset+15*stride];
    double g16 = cart[cart_offset+16*stride];
    double g17 = cart[cart_offset+17*stride];
    double g18 = cart[cart_offset+18*stride];
    double g19 = cart[cart_offset+19*stride];
    double g20 = cart[cart_offset+20*stride];
    double g21 = cart[cart_offset+21*stride];
    double g22 = cart[cart_offset+22*stride];
    double g23 = cart[cart_offset+23*stride];
    double g24 = cart[cart_offset+24*stride];
    double g25 = cart[cart_offset+25*stride];
    double g26 = cart[cart_offset+26*stride];
    double g27 = cart[cart_offset+27*stride];
    double g28 = cart[cart_offset+28*stride];
    double g29 = cart[cart_offset+29*stride];
    double g30 = cart[cart_offset+30*stride];
    double g31 = cart[cart_offset+31*stride];
    double g32 = cart[cart_offset+32*stride];
    double g33 = cart[cart_offset+33*stride];
    double g34 = cart[cart_offset+34*stride];
    double g35 = cart[cart_offset+35*stride];
    sph[sph_offset+0*stride] = 4.9501391276721742 * g1 + -24.7506956383608703 * g6 + 14.8504173830165218 * g15 + -0.7071627325245963 * g28;
    sph[sph_offset+1*stride] = 15.8757639708114002 * (g4 + g22) + -52.9192132360380043 * g11;
    sph[sph_offset+2*stride] = -2.5945778936013020 * (g1 - g6) + 31.1349347232156219 * g8 + 4.6702402084823440 * g15 + -62.2698694464312439 * g17 + -0.5189155787202604 * g28 + 6.2269869446431247 * g30;
    sph[sph_offset+3*stride] = -12.4539738892862495 * (g4 - g22) + 41.5132462976208316 * (g13 - g24);
    sph[sph_offset+4*stride] = 1.4081304047606462 * g1 + 2.3468840079344107 * g6 + -28.1626080952129243 * g8 + 0.4693768015868821 * (g15 - g28) + -18.7750720634752817 * g17 + 37.5501441269505705 * g19 + 9.3875360317376408 * g30 + -12.5167147089835229 * g32;
    sph[sph_offset+5*stride] = 6.6379903866747414 * (g4 + g22) + 13.2759807733494828 * g11 + -35.4026153955986160 * (g13 + g24) + 21.2415692373591725 * g26;
    sph[sph_offset+6*stride] = -0.4516580379125866 * (g1 + g28) + -1.3549741137377600 * (g6 + g15) + 10.8397929099020782 * (g8 + g30) + 21.6795858198041564 * (g17 - g19 - g32) + 5.7812228852811094 * g34;
    sph[sph_offset+7*stride] = -2.3899496919201728 * (g2 + g29) + -7.1698490757605189 * (g7 + g16) + 14.3396981515210360 * (g9 + g31) + 28.6793963030420720 * g18 + -11.4717585212168292 * (g20 + g33) + 1.0925484305920790 * g35;
    sph[sph_offset+8*stride] = -0.4516580379125866 * (g0 + g21) + -1.3549741137377600 * (g3 + g10) + 10.8397929099020782 * (g5 + g23) + 21.6795858198041564 * g12 + -21.6795858198041564 * (g14 + g25) + 5.7812228852811094 * g27;
    sph[sph_offset+9*stride] = 3.3189951933373707 * (g2 + g7 - g16 - g29) + -17.7013076977993080 * (g9 - g31) + 10.6207846186795862 * (g20 - g33);
    sph[sph_offset+10*stride] = 0.4693768015868821 * (g0 - g3) + -9.3875360317376408 * g5 + -2.3468840079344107 * g10 + 18.7750720634752817 * g12 + 12.5167147089835229 * g14 + -1.4081304047606462 * g21 + 28.1626080952129243 * g23 + -37.5501441269505705 * g25;
    sph[sph_offset+11*stride] = -3.1134934723215624 * (g2 + g29) + 15.5674673616078110 * (g7 + g16) + 10.3783115744052079 * (g9 + g31) + -62.2698694464312439 * g18;
    sph[sph_offset+12*stride] = -0.5189155787202604 * g0 + 4.6702402084823440 * g3 + 6.2269869446431247 * g5 + 2.5945778936013020 * (g10 - g21) + -62.2698694464312439 * g12 + 31.1349347232156219 * g23;
    sph[sph_offset+13*stride] = 2.6459606618019000 * (g2 - g29) + -39.6894099270284997 * (g7 - g16);
    sph[sph_offset+14*stride] = 0.7071627325245963 * g0 + -14.8504173830165218 * g3 + 24.7506956383608703 * g10 + -4.9501391276721742 * g21;
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
        case 5: _cart2sph_ang5 <<<blocks, threads, 0, stream>>> (cart_gto, sph_gto, stride, count); break;
        case 6: _cart2sph_ang6 <<<blocks, threads, 0, stream>>> (cart_gto, sph_gto, stride, count); break;
        case 7: _cart2sph_ang7 <<<blocks, threads, 0, stream>>> (cart_gto, sph_gto, stride, count); break;
        default:
            fprintf(stderr, "Ang > 7 is not supported!\n");
            return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
