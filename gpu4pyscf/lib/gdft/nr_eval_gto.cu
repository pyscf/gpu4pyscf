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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "gint/gint.h"
#include "gint/cuda_alloc.cuh"
#include "nr_eval_gto.cuh"
#include "contract_rho.cuh"

#define NG_PER_BLOCK      256
#define LMAX            8

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

template <int ANG> __device__
static void _nabla1(double *fx1, double *fy1, double *fz1,
                    double *fx0, double *fy0, double *fz0, double a){
    double a2 = -2 * a;
    fx1[0] = a2*fx0[1];
    fy1[0] = a2*fy0[1];
    fz1[0] = a2*fz0[1];
#pragma unroll
    for (int i = 1; i <= ANG; i++) {
        fx1[i] = i*fx0[i-1] + a2*fx0[i+1];
        fy1[i] = i*fy0[i-1] + a2*fy0[i+1];
        fz1[i] = i*fz0[i-1] + a2*fz0[i+1];
    }
}

__global__
static void _screen_index(int *non0shl_idx, double cutoff, int ang, int nprim, 
        double *coords, int ngrids, int bas_offset, GTOValEnvVars gto_envs){
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int ish = blockIdx.y + bas_offset;
    const bool active = grid_id < ngrids;

    int natm = gto_envs.natm;
    int atm_id = gto_envs.bas_atom[ish];
    double* atm_coords = gto_envs.atom_coordx;
    double gridx, gridy, gridz;
    if (active) {
        gridx = coords[0*ngrids + grid_id];
        gridy = coords[1*ngrids + grid_id];
        gridz = coords[2*ngrids + grid_id];
    } else {
        gridx = 0.0;
        gridy = 0.0;
        gridz = 0.0;
    }
    double rx = gridx - atm_coords[atm_id + 0*natm];
    double ry = gridy - atm_coords[atm_id + 1*natm];
    double rz = gridz - atm_coords[atm_id + 2*natm];
    double rr = rx * rx + ry * ry + rz * rz;
    double r = sqrt(rr);

    double *exps = gto_envs.env + gto_envs.bas_exp[ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[ish];
    /*
    double maxc = 0.0;
    double min_exp = 1e9;
    for (int ip = 0; ip < nprim; ++ip) {
        min_exp = MIN(min_exp, exps[ip]);
        maxc = MAX(maxc, fabs(coeffs[ip]));
    }
    double gto_sup = -min_exp * rr + .5 * log(rr) * l + log(maxc);
    int is_large = gto_sup > log(cutoff);
    */
    double gto_sup = 0.0;
    for (int ip = 0; ip < nprim; ++ip) {
        gto_sup += coeffs[ip] * exp(-exps[ip] * rr);
    }
    gto_sup *= pow(r,ang);
    int is_large = fabs(gto_sup) > cutoff;

    // Reduce and write to global memory
    unsigned int tx = threadIdx.x;
    __shared__ int sdata[NG_PER_BLOCK];
    sdata[tx] = active ? is_large : 0;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] = sdata[tx] || sdata[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0 && active){
        atomicOr(non0shl_idx + ish, sdata[0]);
    }
}

template <int ANG> __device__
static void _cart2sph(double *g_cart, double *g_sph, int stride, int grid_id){
    if (ANG == 0) {
        g_sph[grid_id           ] += g_cart[0];
    } else if (ANG == 1){
        g_sph[grid_id           ] += g_cart[0];
        g_sph[grid_id +   stride] += g_cart[1];
        g_sph[grid_id + 2*stride] += g_cart[2];
    } else if (ANG == 2){
        g_sph[grid_id           ] += 1.092548430592079070 * g_cart[1];
        g_sph[grid_id +   stride] += 1.092548430592079070 * g_cart[4];
        g_sph[grid_id + 2*stride] += 0.630783130505040012 * g_cart[5] - 0.315391565252520002 * (g_cart[0] + g_cart[3]);
        g_sph[grid_id + 3*stride] += 1.092548430592079070 * g_cart[2];
        g_sph[grid_id + 4*stride] += 0.546274215296039535 * (g_cart[0] - g_cart[3]);
    } else if (ANG == 3){
        g_sph[grid_id           ] += 1.770130769779930531 * g_cart[1] - 0.590043589926643510 * g_cart[6];
        g_sph[grid_id +   stride] += 2.890611442640554055 * g_cart[4];
        g_sph[grid_id + 2*stride] += 1.828183197857862944 * g_cart[8] - 0.457045799464465739 * (g_cart[1] + g_cart[6]);
        g_sph[grid_id + 3*stride] += 0.746352665180230782 * g_cart[9] - 1.119528997770346170 * (g_cart[2] + g_cart[7]);
        g_sph[grid_id + 4*stride] += 1.828183197857862944 * g_cart[5] - 0.457045799464465739 * (g_cart[0] + g_cart[3]);
        g_sph[grid_id + 5*stride] += 1.445305721320277020 * (g_cart[2] - g_cart[7]);
        g_sph[grid_id + 6*stride] += 0.590043589926643510 * g_cart[0] - 1.770130769779930530 * g_cart[3];
    } else if (ANG == 4){
        g_sph[grid_id           ] += 2.503342941796704538 * (g_cart[1] - g_cart[6]) ;
        g_sph[grid_id +   stride] += 5.310392309339791593 * g_cart[4] - 1.770130769779930530 * g_cart[11];
        g_sph[grid_id + 2*stride] += 5.677048174545360108 * g_cart[8] - 0.946174695757560014 * (g_cart[1] + g_cart[6]);
        g_sph[grid_id + 3*stride] += 2.676186174229156671 * g_cart[13]- 2.007139630671867500 * (g_cart[4] + g_cart[11]);
        g_sph[grid_id + 4*stride] += 0.317356640745612911 * (g_cart[0] + g_cart[10]) + 0.634713281491225822 * g_cart[3] - 2.538853125964903290 * (g_cart[5] + g_cart[12]) + 0.846284375321634430 * g_cart[14];
        g_sph[grid_id + 5*stride] += 2.676186174229156671 * g_cart[9] - 2.007139630671867500 * (g_cart[2] + g_cart[7]);
        g_sph[grid_id + 6*stride] += 2.838524087272680054 * (g_cart[5] - g_cart[12]) + 0.473087347878780009 * (g_cart[10]- g_cart[0]);
        g_sph[grid_id + 7*stride] += 1.770130769779930531 * g_cart[2] - 5.310392309339791590 * g_cart[7];
        g_sph[grid_id + 8*stride] += 0.625835735449176134 * (g_cart[0] + g_cart[10]) - 3.755014412695056800 * g_cart[3];
    } else if (ANG == 5) {
        g_sph[grid_id           ] += 3.2819102842008507*g_cart[1] + -6.563820568401701*g_cart[6] + 0.6563820568401701*g_cart[15];
        g_sph[grid_id +   stride] += 8.302649259524165*g_cart[4] + -8.302649259524165*g_cart[11];
        g_sph[grid_id + 2*stride] += -1.467714898305751*g_cart[1] + -0.9784765988705008*g_cart[6] + 11.741719186446009*g_cart[8] + 0.4892382994352504*g_cart[15] + -3.913906395482003*g_cart[17];
        g_sph[grid_id + 3*stride] += -4.793536784973324*g_cart[4] + -4.793536784973324*g_cart[11] + 9.587073569946648*g_cart[13];
        g_sph[grid_id + 4*stride] += 0.45294665119569694*g_cart[1] + 0.9058933023913939*g_cart[6] + -5.435359814348363*g_cart[8] + 0.45294665119569694*g_cart[15] + -5.435359814348363*g_cart[17] + 3.6235732095655755*g_cart[19];
        g_sph[grid_id + 5*stride] += 1.754254836801354*g_cart[2] + 3.508509673602708*g_cart[7] + -4.678012898136944*g_cart[9] + 1.754254836801354*g_cart[16] + -4.678012898136944*g_cart[18] + 0.9356025796273888*g_cart[20];
        g_sph[grid_id + 6*stride] += 0.45294665119569694*g_cart[0] + 0.9058933023913939*g_cart[3] + -5.435359814348363*g_cart[5] + 0.45294665119569694*g_cart[10] + -5.435359814348363*g_cart[12] + 3.6235732095655755*g_cart[14];
        g_sph[grid_id + 7*stride] += -2.396768392486662*g_cart[2] + 4.793536784973324*g_cart[9] + 2.396768392486662*g_cart[16] + -4.793536784973324*g_cart[18];
        g_sph[grid_id + 8*stride] += -0.4892382994352504*g_cart[0] + 0.9784765988705008*g_cart[3] + 3.913906395482003*g_cart[5] + 1.467714898305751*g_cart[10] + -11.741719186446009*g_cart[12];
        g_sph[grid_id + 9*stride] += 2.075662314881041*g_cart[2] + -12.453973889286248*g_cart[7] + 2.075662314881041*g_cart[16];
        g_sph[grid_id +10*stride] += 0.6563820568401701*g_cart[0] + -6.563820568401701*g_cart[3] + 3.2819102842008507*g_cart[10];
        /*
        // Generated by ChatGPT
        g_sph[0]  = (3.2819102842008507 * g_cart[1]) + (-6.563820568401701 * g_cart[6]) + (0.6563820568401701 * g_cart[15]);
        g_sph[1]  = 8.302649259524165 * (g_cart[4] - g_cart[11]);
        g_sph[2]  = (-1.467714898305751 * g_cart[1]) + (-0.9784765988705008 * g_cart[6]) + (11.741719186446009 * g_cart[8]) + (0.4892382994352504 * g_cart[15]) + (-3.913906395482003 * g_cart[17]);
        g_sph[3]  = -4.793536784973324 * (g_cart[4] + g_cart[11]) + 9.587073569946648 * g_cart[13];
        g_sph[4]  = (0.45294665119569694 * (g_cart[1] + g_cart[15])) + (0.9058933023913939 * g_cart[6]) + (-5.435359814348363 * (g_cart[8] + g_cart[17])) + (3.6235732095655755 * g_cart[19]);
        g_sph[5]  = 1.754254836801354 * (g_cart[2] + g_cart[16]) + 3.508509673602708 * g_cart[7] + (-4.678012898136944 * (g_cart[9] + g_cart[18])) + (0.9356025796273888 * g_cart[20]);
        g_sph[6]  = (0.45294665119569694 * (g_cart[0] + g_cart[10])) + (0.9058933023913939 * g_cart[3]) + (-5.435359814348363 * (g_cart[5] + g_cart[12])) + (3.6235732095655755 * g_cart[14]);
        g_sph[7]  = -2.396768392486662 * (g_cart[2] - g_cart[16]) + 4.793536784973324 * (g_cart[9] - g_cart[18]);
        g_sph[8]  = (-0.4892382994352504 * g_cart[0]) + (0.9784765988705008 * g_cart[3]) + (3.913906395482003 * g_cart[5]) + (1.467714898305751 * g_cart[10]) + (-11.741719186446009 * g_cart[12]);
        g_sph[9]  = 2.075662314881041 * (g_cart[2] + g_cart[16]) - 12.453973889286248 * g_cart[7];
        g_sph[10] = (0.6563820568401701 * g_cart[0]) + (-6.563820568401701 * g_cart[3]) + (3.2819102842008507 * g_cart[10]);
        */
    } else if (ANG == 6) {
        g_sph[grid_id           ] += 4.099104631151486*g_cart[1] + -13.663682103838289*g_cart[6] + 4.099104631151486*g_cart[15];
        g_sph[grid_id + 1*stride] += 11.833095811158763*g_cart[4] + -23.666191622317527*g_cart[11] + 2.3666191622317525*g_cart[22];
        g_sph[grid_id + 2*stride] += -2.0182596029148963*g_cart[1] + 20.182596029148968*g_cart[8] + 2.0182596029148963*g_cart[15] + -20.182596029148968*g_cart[17];
        g_sph[grid_id + 3*stride] += -8.29084733563431*g_cart[4] + -5.527231557089541*g_cart[11] + 22.108926228358165*g_cart[13] + 2.7636157785447706*g_cart[22] + -7.369642076119389*g_cart[24];
        g_sph[grid_id + 4*stride] += 0.9212052595149236*g_cart[1] + 1.8424105190298472*g_cart[6] + -14.739284152238778*g_cart[8] + 0.9212052595149236*g_cart[15] + -14.739284152238778*g_cart[17] + 14.739284152238778*g_cart[19];
        g_sph[grid_id + 5*stride] += 2.913106812593657*g_cart[4] + 5.826213625187314*g_cart[11] + -11.652427250374627*g_cart[13] + 2.913106812593657*g_cart[22] + -11.652427250374627*g_cart[24] + 4.6609709001498505*g_cart[26];
        g_sph[grid_id + 6*stride] += -0.3178460113381421*g_cart[0] + -0.9535380340144264*g_cart[3] + 5.721228204086558*g_cart[5] + -0.9535380340144264*g_cart[10] + 11.442456408173117*g_cart[12] + -7.628304272115411*g_cart[14] + -0.3178460113381421*g_cart[21] + 5.721228204086558*g_cart[23] + -7.628304272115411*g_cart[25] + 1.0171072362820548*g_cart[27];
        g_sph[grid_id + 7*stride] += 2.913106812593657*g_cart[2] + 5.826213625187314*g_cart[7] + -11.652427250374627*g_cart[9] + 2.913106812593657*g_cart[16] + -11.652427250374627*g_cart[18] + 4.6609709001498505*g_cart[20];
        g_sph[grid_id + 8*stride] += 0.4606026297574618*g_cart[0] + 0.4606026297574618*g_cart[3] + -7.369642076119389*g_cart[5] + -0.4606026297574618*g_cart[10] + 7.369642076119389*g_cart[14] + -0.4606026297574618*g_cart[21] + 7.369642076119389*g_cart[23] + -7.369642076119389*g_cart[25];
        g_sph[grid_id + 9*stride] += -2.7636157785447706*g_cart[2] + 5.527231557089541*g_cart[7] + 7.369642076119389*g_cart[9] + 8.29084733563431*g_cart[16] + -22.108926228358165*g_cart[18];
        g_sph[grid_id +10*stride] += -0.5045649007287241*g_cart[0] + 2.52282450364362*g_cart[3] + 5.045649007287242*g_cart[5] + 2.52282450364362*g_cart[10] + -30.273894043723452*g_cart[12] + -0.5045649007287241*g_cart[21] + 5.045649007287242*g_cart[23];
        g_sph[grid_id +11*stride] += 2.3666191622317525*g_cart[2] + -23.666191622317527*g_cart[7] + 11.833095811158763*g_cart[16];
        g_sph[grid_id +12*stride] += 0.6831841051919144*g_cart[0] + -10.247761577878716*g_cart[3] + 10.247761577878716*g_cart[10] + -0.6831841051919144*g_cart[21];
        /*
        // Generated by ChatGPT
        g_sph[0]  = 4.099104631151486 * (g_cart[1] + g_cart[15]) - 13.663682103838289 * g_cart[6];
        g_sph[1]  = 11.833095811158763 * (g_cart[4] - 2 * g_cart[11]) + 2.3666191622317525 * g_cart[22];
        g_sph[2]  = -2.0182596029148963 * (g_cart[1] - g_cart[15]) + 20.182596029148968 * (g_cart[8] - g_cart[17]);
        g_sph[3]  = -8.29084733563431 * g_cart[4] - 5.527231557089541 * g_cart[11] + 22.108926228358165 * g_cart[13] + 2.7636157785447706 * g_cart[22] - 7.369642076119389 * g_cart[24];
        g_sph[4]  = 0.9212052595149236 * (g_cart[1] + g_cart[15]) + 1.8424105190298472 * g_cart[6] - 14.739284152238778 * (g_cart[8] + g_cart[17] - g_cart[19]);
        g_sph[5]  = 2.913106812593657 * (g_cart[4] + g_cart[22]) + 5.826213625187314 * g_cart[11] - 11.652427250374627 * (g_cart[13] + g_cart[24]) + 4.6609709001498505 * g_cart[26];
        g_sph[6]  = -0.3178460113381421 * (g_cart[0] + g_cart[21]) - 0.9535380340144264 * (g_cart[3] + g_cart[10]) + 5.721228204086558 * (g_cart[5] + g_cart[23]) + 11.442456408173117 * g_cart[12] - 7.628304272115411 * (g_cart[14] + g_cart[25]) + 1.0171072362820548 * g_cart[27];
        g_sph[7]  = 2.913106812593657 * (g_cart[2] + g_cart[16]) + 5.826213625187314 * g_cart[7] - 11.652427250374627 * (g_cart[9] + g_cart[18]) + 4.6609709001498505 * g_cart[20];
        g_sph[8]  = 0.4606026297574618 * (g_cart[0] + g_cart[3] - g_cart[10] - g_cart[21]) - 7.369642076119389 * (g_cart[5] - g_cart[14] + g_cart[23] - g_cart[25]);
        g_sph[9]  = -2.7636157785447706 * g_cart[2] + 5.527231557089541 * g_cart[7] + 7.369642076119389 * g_cart[9] + 8.29084733563431 * g_cart[16] - 22.108926228358165 * g_cart[18];
        g_sph[10] = -0.5045649007287241 * (g_cart[0] + g_cart[21]) + 5.045649007287242 * (g_cart[5] + g_cart[23]) + 2.52282450364362 * (g_cart[3] + g_cart[10]) - 30.273894043723452 * g_cart[12];
        g_sph[11] = 2.3666191622317525 * (g_cart[2] + g_cart[16]) - 23.666191622317527 * g_cart[7];
        g_sph[12] = 0.6831841051919144 * (g_cart[0] - g_cart[21]) - 10.247761577878716 * (g_cart[3] - g_cart[10]);
        */
    } else if(ANG == 7) {
        g_sph[grid_id           ] += 4.950139127672174*g_cart[1] + -24.75069563836087*g_cart[6] + 14.850417383016522*g_cart[15] + -0.7071627325245963*g_cart[28];
        g_sph[grid_id +   stride] += 15.8757639708114*g_cart[4] + -52.919213236038004*g_cart[11] + 15.8757639708114*g_cart[22];
        g_sph[grid_id + 2*stride] += -2.594577893601302*g_cart[1] + 2.594577893601302*g_cart[6] + 31.134934723215622*g_cart[8] + 4.670240208482344*g_cart[15] + -62.269869446431244*g_cart[17] + -0.5189155787202604*g_cart[28] + 6.226986944643125*g_cart[30];
        g_sph[grid_id + 3*stride] += -12.45397388928625*g_cart[4] + 41.51324629762083*g_cart[13] + 12.45397388928625*g_cart[22] + -41.51324629762083*g_cart[24];
        g_sph[grid_id + 4*stride] += 1.4081304047606462*g_cart[1] + 2.3468840079344107*g_cart[6] + -28.162608095212924*g_cart[8] + 0.4693768015868821*g_cart[15] + -18.77507206347528*g_cart[17] + 37.55014412695057*g_cart[19] + -0.4693768015868821*g_cart[28] + 9.38753603173764*g_cart[30] + -12.516714708983523*g_cart[32];
        g_sph[grid_id + 5*stride] += 6.637990386674741*g_cart[4] + 13.275980773349483*g_cart[11] + -35.402615395598616*g_cart[13] + 6.637990386674741*g_cart[22] + -35.402615395598616*g_cart[24] + 21.241569237359172*g_cart[26];
        g_sph[grid_id + 6*stride] += -0.4516580379125866*g_cart[1] + -1.35497411373776*g_cart[6] + 10.839792909902078*g_cart[8] + -1.35497411373776*g_cart[15] + 21.679585819804156*g_cart[17] + -21.679585819804156*g_cart[19] + -0.4516580379125866*g_cart[28] + 10.839792909902078*g_cart[30] + -21.679585819804156*g_cart[32] + 5.781222885281109*g_cart[34];
        g_sph[grid_id + 7*stride] += -2.389949691920173*g_cart[2] + -7.169849075760519*g_cart[7] + 14.339698151521036*g_cart[9] + -7.169849075760519*g_cart[16] + 28.679396303042072*g_cart[18] + -11.47175852121683*g_cart[20] + -2.389949691920173*g_cart[29] + 14.339698151521036*g_cart[31] + -11.47175852121683*g_cart[33] + 1.092548430592079*g_cart[35];
        g_sph[grid_id + 8*stride] += -0.4516580379125866*g_cart[0] + -1.35497411373776*g_cart[3] + 10.839792909902078*g_cart[5] + -1.35497411373776*g_cart[10] + 21.679585819804156*g_cart[12] + -21.679585819804156*g_cart[14] + -0.4516580379125866*g_cart[21] + 10.839792909902078*g_cart[23] + -21.679585819804156*g_cart[25] + 5.781222885281109*g_cart[27];
        g_sph[grid_id + 9*stride] += 3.3189951933373707*g_cart[2] + 3.3189951933373707*g_cart[7] + -17.701307697799308*g_cart[9] + -3.3189951933373707*g_cart[16] + 10.620784618679586*g_cart[20] + -3.3189951933373707*g_cart[29] + 17.701307697799308*g_cart[31] + -10.620784618679586*g_cart[33];
        g_sph[grid_id +10*stride] += 0.4693768015868821*g_cart[0] + -0.4693768015868821*g_cart[3] + -9.38753603173764*g_cart[5] + -2.3468840079344107*g_cart[10] + 18.77507206347528*g_cart[12] + 12.516714708983523*g_cart[14] + -1.4081304047606462*g_cart[21] + 28.162608095212924*g_cart[23] + -37.55014412695057*g_cart[25];
        g_sph[grid_id +11*stride] += -3.1134934723215624*g_cart[2] + 15.567467361607811*g_cart[7] + 10.378311574405208*g_cart[9] + 15.567467361607811*g_cart[16] + -62.269869446431244*g_cart[18] + -3.1134934723215624*g_cart[29] + 10.378311574405208*g_cart[31];
        g_sph[grid_id +12*stride] += -0.5189155787202604*g_cart[0] + 4.670240208482344*g_cart[3] + 6.226986944643125*g_cart[5] + 2.594577893601302*g_cart[10] + -62.269869446431244*g_cart[12] + -2.594577893601302*g_cart[21] + 31.134934723215622*g_cart[23];
        g_sph[grid_id +13*stride] += 2.6459606618019*g_cart[2] + -39.6894099270285*g_cart[7] + 39.6894099270285*g_cart[16] + -2.6459606618019*g_cart[29];
        g_sph[grid_id +14*stride] += 0.7071627325245963*g_cart[0] + -14.850417383016522*g_cart[3] + 24.75069563836087*g_cart[10] + -4.950139127672174*g_cart[21];
        /*
        // Generated by ChatGPT
        g_sph[0]  = 4.950139127672174 * g_cart[1] - 24.75069563836087 * g_cart[6] + 14.850417383016522 * g_cart[15] - 0.7071627325245963 * g_cart[28];
        g_sph[1]  = 15.8757639708114 * (g_cart[4] + g_cart[22]) - 52.919213236038004 * g_cart[11];
        g_sph[2]  = (-2.594577893601302 * (g_cart[1] - g_cart[6])) + (31.134934723215622 * g_cart[8]) + (4.670240208482344 * g_cart[15]) - (62.269869446431244 * g_cart[17]) - (0.5189155787202604 * g_cart[28]) + (6.226986944643125 * g_cart[30]);
        g_sph[3]  = -12.45397388928625 * (g_cart[4] - g_cart[22]) + 41.51324629762083 * (g_cart[13] - g_cart[24]);
        g_sph[4]  = (1.4081304047606462 * g_cart[1]) + (2.3468840079344107 * g_cart[6]) - (28.162608095212924 * g_cart[8]) + (0.4693768015868821 * (g_cart[15] - g_cart[28])) - (18.77507206347528 * g_cart[17]) + (37.55014412695057 * g_cart[19]) + (9.38753603173764 * g_cart[30]) - (12.516714708983523 * g_cart[32]);
        g_sph[5]  = 6.637990386674741 * (g_cart[4] + g_cart[22]) + 13.275980773349483 * g_cart[11] - 35.402615395598616 * (g_cart[13] + g_cart[24]) + 21.241569237359172 * g_cart[26];
        g_sph[6]  = (-0.4516580379125866 * (g_cart[1] + g_cart[28])) + (-1.35497411373776 * (g_cart[6] + g_cart[15])) + (10.839792909902078 * g_cart[8]) + (21.679585819804156 * (g_cart[17] - g_cart[19])) + (10.839792909902078 * g_cart[30]) - (21.679585819804156 * g_cart[32]) + (5.781222885281109 * g_cart[34]);
        g_sph[7]  = -2.389949691920173 * (g_cart[2] + g_cart[29]) - 7.169849075760519 * (g_cart[7] + g_cart[16]) + 14.339698151521036 * g_cart[9] + 28.679396303042072 * g_cart[18] - 11.47175852121683 * (g_cart[20] + g_cart[33]) + (1.092548430592079 * g_cart[35]);
        g_sph[8]  = (-0.4516580379125866 * (g_cart[0] + g_cart[21])) + (-1.35497411373776 * (g_cart[3] + g_cart[10])) + (10.839792909902078 * g_cart[5]) + (21.679585819804156 * (g_cart[12] - g_cart[14])) + (10.839792909902078 * g_cart[23]) - (21.679585819804156 * g_cart[25]) + (5.781222885281109 * g_cart[27]);
        g_sph[9]  = 3.3189951933373707 * (g_cart[2] + g_cart[7] - g_cart[16] - g_cart[29]) - 17.701307697799308 * g_cart[9] + 10.620784618679586 * (g_cart[20] - g_cart[33]) + 17.701307697799308 * g_cart[31];
        g_sph[10] = (0.4693768015868821 * (g_cart[0] - g_cart[3])) - (9.38753603173764 * g_cart[5]) - (2.3468840079344107 * g_cart[10]) + (18.77507206347528 * g_cart[12]) + (12.516714708983523 * g_cart[14]) - (1.4081304047606462 * g_cart[21]) + (28.162608095212924 * g_cart[23]) - (37.55014412695057 * g_cart[25]);
        g_sph[11] = (-3.1134934723215624 * (g_cart[2] + g_cart[29])) + (15.567467361607811 * (g_cart[7] + g_cart[16])) + (10.378311574405208 * g_cart[9]) - (62.269869446431244 * g_cart[18]) + (10.378311574405208 * g_cart[31]);
        g_sph[12] = (-0.5189155787202604 * g_cart[0]) + (4.670240208482344 * g_cart[3]) + (6.226986944643125 * g_cart[5]) + (2.594577893601302 * g_cart[10]) - (62.269869446431244 * g_cart[12]) - (2.594577893601302 * g_cart[21]) + (31.134934723215622 * g_cart[23]);
        g_sph[13] = (2.6459606618019 * (g_cart[2] - g_cart[29])) - 39.6894099270285 * (g_cart[7] - g_cart[16]);
        g_sph[14] = (0.7071627325245963 * g_cart[0]) - (14.850417383016522 * g_cart[3]) + (24.75069563836087 * g_cart[10]) - (4.950139127672174 * g_cart[21]);
        */
    } else if(ANG == 8){
        g_sph[grid_id           ] += 5.83141328139864*g_cart[1] + -40.81989296979048*g_cart[6] + 40.81989296979048*g_cart[15] + -5.83141328139864*g_cart[28];
        g_sph[grid_id +   stride] += 20.40994648489524*g_cart[4] + -102.0497324244762*g_cart[11] + 61.22983945468572*g_cart[22] + -2.91570664069932*g_cart[37];
        g_sph[grid_id + 2*stride] += -3.193996596357255*g_cart[1] + 7.452658724833595*g_cart[6] + 44.71595234900157*g_cart[8] + 7.452658724833595*g_cart[15] + -149.0531744966719*g_cart[17] + -3.193996596357255*g_cart[28] + 44.71595234900157*g_cart[30];
        g_sph[grid_id + 3*stride] += -17.24955311049054*g_cart[4] + 17.24955311049054*g_cart[11] + 68.99821244196217*g_cart[13] + 31.04919559888297*g_cart[22] + -137.9964248839243*g_cart[24] + -3.449910622098108*g_cart[37] + 13.79964248839243*g_cart[39];
        g_sph[grid_id + 4*stride] += 1.913666099037323*g_cart[1] + 1.913666099037323*g_cart[6] + -45.92798637689575*g_cart[8] + -1.913666099037323*g_cart[15] + 76.54664396149292*g_cart[19] + -1.913666099037323*g_cart[28] + 45.92798637689575*g_cart[30] + -76.54664396149292*g_cart[32];
        g_sph[grid_id + 5*stride] += 11.1173953976599*g_cart[4] + 18.52899232943316*g_cart[11] + -74.11596931773265*g_cart[13] + 3.705798465886632*g_cart[22] + -49.41064621182176*g_cart[24] + 59.29277545418611*g_cart[26] + -3.705798465886632*g_cart[37] + 24.70532310591088*g_cart[39] + -19.7642584847287*g_cart[41];
        g_sph[grid_id + 6*stride] += -0.912304516869819*g_cart[1] + -2.736913550609457*g_cart[6] + 27.36913550609457*g_cart[8] + -2.736913550609457*g_cart[15] + 54.73827101218914*g_cart[17] + -72.98436134958553*g_cart[19] + -0.912304516869819*g_cart[28] + 27.36913550609457*g_cart[30] + -72.98436134958553*g_cart[32] + 29.19374453983421*g_cart[34];
        g_sph[grid_id + 7*stride] += -3.8164436064573*g_cart[4] + -11.4493308193719*g_cart[11] + 30.5315488516584*g_cart[13] + -11.4493308193719*g_cart[22] + 61.06309770331679*g_cart[24] + -36.63785862199007*g_cart[26] + -3.8164436064573*g_cart[37] + 30.5315488516584*g_cart[39] + -36.63785862199007*g_cart[41] + 6.978639737521918*g_cart[43];
        g_sph[grid_id + 8*stride] += 0.3180369672047749*g_cart[0] + 1.272147868819099*g_cart[3] + -10.1771829505528*g_cart[5] + 1.908221803228649*g_cart[10] + -30.53154885165839*g_cart[12] + 30.53154885165839*g_cart[14] + 1.272147868819099*g_cart[21] + -30.53154885165839*g_cart[23] + 61.06309770331677*g_cart[25] + -16.28349272088447*g_cart[27] + 0.3180369672047749*g_cart[36] + -10.1771829505528*g_cart[38] + 30.53154885165839*g_cart[40] + -16.28349272088447*g_cart[42] + 1.16310662292032*g_cart[44];
        g_sph[grid_id + 9*stride] += -3.8164436064573*g_cart[2] + -11.4493308193719*g_cart[7] + 30.5315488516584*g_cart[9] + -11.4493308193719*g_cart[16] + 61.06309770331679*g_cart[18] + -36.63785862199007*g_cart[20] + -3.8164436064573*g_cart[29] + 30.5315488516584*g_cart[31] + -36.63785862199007*g_cart[33] + 6.978639737521918*g_cart[35];
        g_sph[grid_id +10*stride] += -0.4561522584349095*g_cart[0] + -0.912304516869819*g_cart[3] + 13.68456775304729*g_cart[5] + 13.68456775304729*g_cart[12] + -36.49218067479276*g_cart[14] + 0.912304516869819*g_cart[21] + -13.68456775304729*g_cart[23] + 14.5968722699171*g_cart[27] + 0.4561522584349095*g_cart[36] + -13.68456775304729*g_cart[38] + 36.49218067479276*g_cart[40] + -14.5968722699171*g_cart[42];
        g_sph[grid_id +11*stride] += 3.705798465886632*g_cart[2] + -3.705798465886632*g_cart[7] + -24.70532310591088*g_cart[9] + -18.52899232943316*g_cart[16] + 49.41064621182176*g_cart[18] + 19.7642584847287*g_cart[20] + -11.1173953976599*g_cart[29] + 74.11596931773265*g_cart[31] + -59.29277545418611*g_cart[33];
        g_sph[grid_id +12*stride] += 0.4784165247593308*g_cart[0] + -1.913666099037323*g_cart[3] + -11.48199659422394*g_cart[5] + -4.784165247593307*g_cart[10] + 57.40998297111968*g_cart[12] + 19.13666099037323*g_cart[14] + -1.913666099037323*g_cart[21] + 57.40998297111968*g_cart[23] + -114.8199659422394*g_cart[25] + 0.4784165247593308*g_cart[36] + -11.48199659422394*g_cart[38] + 19.13666099037323*g_cart[40];
        g_sph[grid_id +13*stride] += -3.449910622098108*g_cart[2] + 31.04919559888297*g_cart[7] + 13.79964248839243*g_cart[9] + 17.24955311049054*g_cart[16] + -137.9964248839243*g_cart[18] + -17.24955311049054*g_cart[29] + 68.99821244196217*g_cart[31];
        g_sph[grid_id +14*stride] += -0.5323327660595425*g_cart[0] + 7.452658724833595*g_cart[3] + 7.452658724833595*g_cart[5] + -111.7898808725039*g_cart[12] + -7.452658724833595*g_cart[21] + 111.7898808725039*g_cart[23] + 0.5323327660595425*g_cart[36] + -7.452658724833595*g_cart[38];
        g_sph[grid_id +15*stride] += 2.91570664069932*g_cart[2] + -61.22983945468572*g_cart[7] + 102.0497324244762*g_cart[16] + -20.40994648489524*g_cart[29];
        g_sph[grid_id +16*stride] += 0.72892666017483*g_cart[0] + -20.40994648489524*g_cart[3] + 51.0248662122381*g_cart[10] + -20.40994648489524*g_cart[21] + 0.72892666017483*g_cart[36];
        /*
        // Generated by ChatGPT
        g_sph[0]  = 5.83141328139864 * (g_cart[1] - g_cart[28]) + 40.81989296979048 * (g_cart[15] - g_cart[6]);
        g_sph[1]  = 20.40994648489524 * (g_cart[4] - 5 * g_cart[11]) + 61.22983945468572 * g_cart[22] - 2.91570664069932 * g_cart[37];
        g_sph[2]  = -3.193996596357255 * (g_cart[1] + g_cart[28]) + 7.452658724833595 * (g_cart[6] + g_cart[15]) + 44.71595234900157 * (g_cart[8] + g_cart[30]) - 149.0531744966719 * g_cart[17];
        g_sph[3]  = -17.24955311049054 * (g_cart[4] - g_cart[11]) + 68.99821244196217 * g_cart[13] + 31.04919559888297 * g_cart[22] - 137.9964248839243 * g_cart[24] - 3.449910622098108 * g_cart[37] + 13.79964248839243 * g_cart[39];
        g_sph[4]  = 1.913666099037323 * (g_cart[1] + g_cart[6] - g_cart[15] - g_cart[28]) - 45.92798637689575 * g_cart[8] + 76.54664396149292 * (g_cart[19] - g_cart[32]) + 45.92798637689575 * g_cart[30];
        g_sph[5]  = 11.1173953976599 * g_cart[4] + 18.52899232943316 * g_cart[11] - 74.11596931773265 * g_cart[13] + 3.705798465886632 * (g_cart[22] - g_cart[37]) - 49.41064621182176 * g_cart[24] + 59.29277545418611 * g_cart[26] + 24.70532310591088 * g_cart[39] - 19.7642584847287 * g_cart[41];
        g_sph[6]  = -0.912304516869819 * (g_cart[1] + g_cart[28]) - 2.736913550609457 * (g_cart[6] + g_cart[15]) + 27.36913550609457 * (g_cart[8] + g_cart[30]) + 54.73827101218914 * g_cart[17] - 72.98436134958553 * g_cart[19] - 72.98436134958553 * g_cart[32] + 29.19374453983421 * g_cart[34];
        g_sph[7]  = -3.8164436064573 * (g_cart[4] + g_cart[37]) - 11.4493308193719 * (g_cart[11] + g_cart[22]) + 30.5315488516584 * (g_cart[13] + g_cart[39]) + 61.06309770331679 * g_cart[24] - 36.63785862199007 * (g_cart[26] + g_cart[41]) + 6.978639737521918 * g_cart[43];
        g_sph[8]  = 0.3180369672047749 * (g_cart[0] + g_cart[36]) + 1.272147868819099 * (g_cart[3] + g_cart[21]) - 10.1771829505528 * (g_cart[5] + g_cart[38]) + 1.908221803228649 * g_cart[10] - 30.53154885165839 * (g_cart[12] - g_cart[14] + g_cart[23] - g_cart[25] + g_cart[40]) + 61.06309770331677 * g_cart[25] - 16.28349272088447 * (g_cart[27] + g_cart[42]) + 1.16310662292032 * g_cart[44];
        g_sph[9]  = -3.8164436064573 * (g_cart[2] + g_cart[29]) - 11.4493308193719 * (g_cart[7] + g_cart[16]) + 30.5315488516584 * g_cart[9] + 61.06309770331679 * g_cart[18] - 36.63785862199007 * (g_cart[20] + g_cart[33]) + 6.978639737521918 * g_cart[35];
        g_sph[10] = -0.4561522584349095 * (g_cart[0] + g_cart[36]) - 0.912304516869819 * (g_cart[3] - g_cart[21]) + 13.68456775304729 * (g_cart[5] + g_cart[12] - g_cart[23] - g_cart[38]) - 36.49218067479276 * g_cart[14] + 14.5968722699171 * (g_cart[27] - g_cart[42]);
        g_sph[11] = 3.705798465886632 * (g_cart[2] - g_cart[7]) - 24.70532310591088 * g_cart[9] - 18.52899232943316 * g_cart[16] + 49.41064621182176 * g_cart[18] + 19.7642584847287 * g_cart[20] - 11.1173953976599 * g_cart[29] + 74.11596931773265 * g_cart[31] - 59.29277545418611 * g_cart[33];
        g_sph[12] = 0.4784165247593308 * (g_cart[0] + g_cart[36]) - 1.913666099037323 * (g_cart[3] + g_cart[21]) - 11.48199659422394 * (g_cart[5] + g_cart[38]) - 4.784165247593307 * g_cart[10] + 57.40998297111968 * (g_cart[12] + g_cart[23]) + 19.13666099037323 * (g_cart[14] + g_cart[40]) - 114.8199659422394 * g_cart[25];
        g_sph[13] = -3.449910622098108 * (g_cart[2] - g_cart[29]) + 31.04919559888297 * g_cart[7] + 13.79964248839243 * g_cart[9] + 17.24955311049054 * g_cart[16] - 137.9964248839243 * g_cart[18] + 68.99821244196217 * g_cart[31];
        g_sph[14] = -0.5323327660595425 * (g_cart[0] + g_cart[36]) + 7.452658724833595 * (g_cart[3] + g_cart[5] - g_cart[21] - g_cart[38]) - 111.7898808725039 * (g_cart[12] - g_cart[23]);
        g_sph[15] = 2.91570664069932 * g_cart[2] - 61.22983945468572 * g_cart[7] + 102.0497324244762 * g_cart[16] - 20.40994648489524 * g_cart[29];
        g_sph[16] = 0.72892666017483 * (g_cart[0] + g_cart[36]) - 20.40994648489524 * (g_cart[3] + g_cart[21]) + 51.0248662122381 * g_cart[10];
        */
    }
}

template <int ANG> __device__
static void _memset_cart(double *g_cart, int count, int ngrids, int nao){
    // Set g[:,:,grid_id] = 0
    for (int deriv = 0; deriv < count; deriv++){
        for (int i = 0; i < (ANG+1)*(ANG+2)/2; i++){
            g_cart[i * ngrids] = 0.0;
        }
        g_cart += nao * ngrids;
    }
}

template <int ANG> __device__
static void _memset_sph(double *g_sph, int count, int ngrids, int nao){
    for (int deriv = 0; deriv < count; deriv++){
        for (int i = 0; i < 2*ANG+1; i++){
            g_sph[i * ngrids] = 0.0;
        }
        g_sph += nao * ngrids;
    }
}

template <int ANG> __device__
static void _cart_gto(double *g, double ce, double *fx, double *fy, double *fz){
    for (int lx = ANG, i = 0; lx >= 0; lx--){
        for (int ly = ANG - lx; ly >= 0; ly--, i++){
            int lz = ANG - lx - ly;
            g[i] = ce * fx[lx] * fy[ly] * fz[lz];
        }
    }
}

template <int ANG> __global__
static void _cart_kernel_deriv0(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto = offsets.data + i0 * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double ce = 0;
    for (int ip = 0; ip < offsets.nprim; ++ip) {
        ce += coeffs[ip] * exp(-exps[ip] * rr);
    }
    ce *= offsets.fac;

    if (ANG == 0) {
        gto[grid_id] = ce;
    } else if (ANG == 1) {
        gto[         grid_id] = ce * rx;
        gto[1*ngrids+grid_id] = ce * ry;
        gto[2*ngrids+grid_id] = ce * rz;
    } else if (ANG == 2) {
        gto[         grid_id] = ce * rx * rx;
        gto[1*ngrids+grid_id] = ce * rx * ry;
        gto[2*ngrids+grid_id] = ce * rx * rz;
        gto[3*ngrids+grid_id] = ce * ry * ry;
        gto[4*ngrids+grid_id] = ce * ry * rz;
        gto[5*ngrids+grid_id] = ce * rz * rz;
    } else if (ANG == 3) {
        gto[         grid_id] = ce * rx * rx * rx;
        gto[1*ngrids+grid_id] = ce * rx * rx * ry;
        gto[2*ngrids+grid_id] = ce * rx * rx * rz;
        gto[3*ngrids+grid_id] = ce * rx * ry * ry;
        gto[4*ngrids+grid_id] = ce * rx * ry * rz;
        gto[5*ngrids+grid_id] = ce * rx * rz * rz;
        gto[6*ngrids+grid_id] = ce * ry * ry * ry;
        gto[7*ngrids+grid_id] = ce * ry * ry * rz;
        gto[8*ngrids+grid_id] = ce * ry * rz * rz;
        gto[9*ngrids+grid_id] = ce * rz * rz * rz;
    } else if (ANG == 4) {
        gto[          grid_id] = ce * rx * rx * rx * rx;
        gto[1 *ngrids+grid_id] = ce * rx * rx * rx * ry;
        gto[2 *ngrids+grid_id] = ce * rx * rx * rx * rz;
        gto[3 *ngrids+grid_id] = ce * rx * rx * ry * ry;
        gto[4 *ngrids+grid_id] = ce * rx * rx * ry * rz;
        gto[5 *ngrids+grid_id] = ce * rx * rx * rz * rz;
        gto[6 *ngrids+grid_id] = ce * rx * ry * ry * ry;
        gto[7 *ngrids+grid_id] = ce * rx * ry * ry * rz;
        gto[8 *ngrids+grid_id] = ce * rx * ry * rz * rz;
        gto[9 *ngrids+grid_id] = ce * rx * rz * rz * rz;
        gto[10*ngrids+grid_id] = ce * ry * ry * ry * ry;
        gto[11*ngrids+grid_id] = ce * ry * ry * ry * rz;
        gto[12*ngrids+grid_id] = ce * ry * ry * rz * rz;
        gto[13*ngrids+grid_id] = ce * ry * rz * rz * rz;
        gto[14*ngrids+grid_id] = ce * rz * rz * rz * rz;
    } else {
        int lx, ly, lz;
        double xpows[ANG+1];
        double ypows[ANG+1];
        double zpows[ANG+1];

        xpows[0] = 1.0;
        ypows[0] = 1.0;
        zpows[0] = 1.0;

        for(lx = 1; lx <= ANG ; lx++){
            xpows[lx] = xpows[lx-1] * rx;
            ypows[lx] = ypows[lx-1] * ry;
            zpows[lx] = zpows[lx-1] * rz;
        }
        for(int i = 0, lx = ANG; lx >= 0; lx--){
            for(ly = ANG - lx; ly >= 0; ly--, i++){
                lz = ANG - lx - ly;
                gto[i*ngrids + grid_id] = xpows[lx] * ypows[ly] * zpows[lz] * ce;
            }
        }
    }
}

template <int ANG> __global__
static void _cart_kernel_deriv1(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto = offsets.data + i0 * ngrids;
    double* __restrict__ gtox = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz = offsets.data + (nao * 3 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double ce = 0;
    double ce_2a = 0;
    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double c = coeffs[ip];
        double exp_ip = exps[ip];
        double e = exp(-exp_ip * rr);
        ce += c * e;
        ce_2a += c * e * exp_ip;
    }
    ce *= offsets.fac;
    ce_2a *= -2 * offsets.fac;

    if (ANG == 0) {
        gto [grid_id] = ce;
        gtox[grid_id] = ce_2a * rx;
        gtoy[grid_id] = ce_2a * ry;
        gtoz[grid_id] = ce_2a * rz;
    }
    else if (ANG == 1) {
        gto [         grid_id] = ce * rx;
        gto [1*ngrids+grid_id] = ce * ry;
        gto [2*ngrids+grid_id] = ce * rz;
        double ax = ce_2a * rx;
        gtox[         grid_id] = ax * rx + ce;
        gtox[1*ngrids+grid_id] = ax * ry;
        gtox[2*ngrids+grid_id] = ax * rz;
        double ay = ce_2a * ry;
        gtoy[         grid_id] = ay * rx;
        gtoy[1*ngrids+grid_id] = ay * ry + ce;
        gtoy[2*ngrids+grid_id] = ay * rz;
        double az = ce_2a * rz;
        gtoz[         grid_id] = az * rx;
        gtoz[1*ngrids+grid_id] = az * ry;
        gtoz[2*ngrids+grid_id] = az * rz + ce;
    }else if (ANG == 2) {
        gto [         grid_id] = ce * rx * rx;
        gto [1*ngrids+grid_id] = ce * rx * ry;
        gto [2*ngrids+grid_id] = ce * rx * rz;
        gto [3*ngrids+grid_id] = ce * ry * ry;
        gto [4*ngrids+grid_id] = ce * ry * rz;
        gto [5*ngrids+grid_id] = ce * rz * rz;
        double ax = ce_2a * rx;
        gtox[         grid_id] = (ax * rx + 2 * ce) * rx;
        gtox[1*ngrids+grid_id] = (ax * rx +     ce) * ry;
        gtox[2*ngrids+grid_id] = (ax * rx +     ce) * rz;
        gtox[3*ngrids+grid_id] = ax * ry * ry;
        gtox[4*ngrids+grid_id] = ax * ry * rz;
        gtox[5*ngrids+grid_id] = ax * rz * rz;
        double ay = ce_2a * ry;
        gtoy[         grid_id] = ay * rx * rx;
        gtoy[1*ngrids+grid_id] = (ay * ry +     ce) * rx;
        gtoy[2*ngrids+grid_id] = ay * rx * rz;
        gtoy[3*ngrids+grid_id] = (ay * ry + 2 * ce) * ry;
        gtoy[4*ngrids+grid_id] = (ay * ry +     ce) * rz;
        gtoy[5*ngrids+grid_id] = ay * rz * rz;
        double az = ce_2a * rz;
        gtoz[         grid_id] = az * rx * rx;
        gtoz[1*ngrids+grid_id] = az * rx * ry;
        gtoz[2*ngrids+grid_id] = (az * rz +     ce) * rx;
        gtoz[3*ngrids+grid_id] = az * ry * ry;
        gtoz[4*ngrids+grid_id] = (az * rz +     ce) * ry;
        gtoz[5*ngrids+grid_id] = (az * rz + 2 * ce) * rz;
    } else if (ANG == 3) {
        gto [         grid_id] = ce * rx * rx * rx;
        gto [1*ngrids+grid_id] = ce * rx * rx * ry;
        gto [2*ngrids+grid_id] = ce * rx * rx * rz;
        gto [3*ngrids+grid_id] = ce * rx * ry * ry;
        gto [4*ngrids+grid_id] = ce * rx * ry * rz;
        gto [5*ngrids+grid_id] = ce * rx * rz * rz;
        gto [6*ngrids+grid_id] = ce * ry * ry * ry;
        gto [7*ngrids+grid_id] = ce * ry * ry * rz;
        gto [8*ngrids+grid_id] = ce * ry * rz * rz;
        gto [9*ngrids+grid_id] = ce * rz * rz * rz;
        double ax = ce_2a * rx;
        gtox[         grid_id] = (ax * rx + 3 * ce) * rx * rx;
        gtox[1*ngrids+grid_id] = (ax * rx + 2 * ce) * rx * ry;
        gtox[2*ngrids+grid_id] = (ax * rx + 2 * ce) * rx * rz;
        gtox[3*ngrids+grid_id] = (ax * rx +     ce) * ry * ry;
        gtox[4*ngrids+grid_id] = (ax * rx +     ce) * ry * rz;
        gtox[5*ngrids+grid_id] = (ax * rx +     ce) * rz * rz;
        gtox[6*ngrids+grid_id] = ax * ry * ry * ry;
        gtox[7*ngrids+grid_id] = ax * ry * ry * rz;
        gtox[8*ngrids+grid_id] = ax * ry * rz * rz;
        gtox[9*ngrids+grid_id] = ax * rz * rz * rz;
        double ay = ce_2a * ry;
        gtoy[         grid_id] = ay * rx * rx * rx;
        gtoy[1*ngrids+grid_id] = (ay * ry +     ce) * rx * rx;
        gtoy[2*ngrids+grid_id] = ay * rx * rx * rz;
        gtoy[3*ngrids+grid_id] = (ay * ry + 2 * ce) * rx * ry;
        gtoy[4*ngrids+grid_id] = (ay * ry +     ce) * rx * rz;
        gtoy[5*ngrids+grid_id] = ay * rx * rz * rz;
        gtoy[6*ngrids+grid_id] = (ay * ry + 3 * ce) * ry * ry;
        gtoy[7*ngrids+grid_id] = (ay * ry + 2 * ce) * ry * rz;
        gtoy[8*ngrids+grid_id] = (ay * ry +     ce) * rz * rz;
        gtoy[9*ngrids+grid_id] = ay * rz * rz * rz;
        double az = ce_2a * rz;
        gtoz[         grid_id] = az * rx * rx * rx;
        gtoz[1*ngrids+grid_id] = az * rx * rx * ry;
        gtoz[2*ngrids+grid_id] = (az * rz +     ce) * rx * rx;
        gtoz[3*ngrids+grid_id] = az * rx * ry * ry;
        gtoz[4*ngrids+grid_id] = (az * rz +     ce) * rx * ry;
        gtoz[5*ngrids+grid_id] = (az * rz + 2 * ce) * rx * rz;
        gtoz[6*ngrids+grid_id] = az * ry * ry * ry;
        gtoz[7*ngrids+grid_id] = (az * rz +     ce) * ry * ry;
        gtoz[8*ngrids+grid_id] = (az * rz + 2 * ce) * ry * rz;
        gtoz[9*ngrids+grid_id] = (az * rz + 3 * ce) * rz * rz;
    }
    else if (ANG == 4) {
        double ax = ce_2a * rx;
        double ay = ce_2a * ry;
        double az = ce_2a * rz;
        double bxxx = ce * rx * rx * rx;
        double bxxy = ce * rx * rx * ry;
        double bxxz = ce * rx * rx * rz;
        double bxyy = ce * rx * ry * ry;
        double bxyz = ce * rx * ry * rz;
        double bxzz = ce * rx * rz * rz;
        double byyy = ce * ry * ry * ry;
        double byyz = ce * ry * ry * rz;
        double byzz = ce * ry * rz * rz;
        double bzzz = ce * rz * rz * rz;
        gto [          grid_id] = ce * rx * rx * rx * rx;
        gto [1 *ngrids+grid_id] = ce * rx * rx * rx * ry;
        gto [2 *ngrids+grid_id] = ce * rx * rx * rx * rz;
        gto [3 *ngrids+grid_id] = ce * rx * rx * ry * ry;
        gto [4 *ngrids+grid_id] = ce * rx * rx * ry * rz;
        gto [5 *ngrids+grid_id] = ce * rx * rx * rz * rz;
        gto [6 *ngrids+grid_id] = ce * rx * ry * ry * ry;
        gto [7 *ngrids+grid_id] = ce * rx * ry * ry * rz;
        gto [8 *ngrids+grid_id] = ce * rx * ry * rz * rz;
        gto [9 *ngrids+grid_id] = ce * rx * rz * rz * rz;
        gto [10*ngrids+grid_id] = ce * ry * ry * ry * ry;
        gto [11*ngrids+grid_id] = ce * ry * ry * ry * rz;
        gto [12*ngrids+grid_id] = ce * ry * ry * rz * rz;
        gto [13*ngrids+grid_id] = ce * ry * rz * rz * rz;
        gto [14*ngrids+grid_id] = ce * rz * rz * rz * rz;
        gtox[          grid_id] = ax * rx * rx * rx * rx + 4 * bxxx;
        gtox[1 *ngrids+grid_id] = ax * rx * rx * rx * ry + 3 * bxxy;
        gtox[2 *ngrids+grid_id] = ax * rx * rx * rx * rz + 3 * bxxz;
        gtox[3 *ngrids+grid_id] = ax * rx * rx * ry * ry + 2 * bxyy;
        gtox[4 *ngrids+grid_id] = ax * rx * rx * ry * rz + 2 * bxyz;
        gtox[5 *ngrids+grid_id] = ax * rx * rx * rz * rz + 2 * bxzz;
        gtox[6 *ngrids+grid_id] = ax * rx * ry * ry * ry +     byyy;
        gtox[7 *ngrids+grid_id] = ax * rx * ry * ry * rz +     byyz;
        gtox[8 *ngrids+grid_id] = ax * rx * ry * rz * rz +     byzz;
        gtox[9 *ngrids+grid_id] = ax * rx * rz * rz * rz +     bzzz;
        gtox[10*ngrids+grid_id] = ax * ry * ry * ry * ry;
        gtox[11*ngrids+grid_id] = ax * ry * ry * ry * rz;
        gtox[12*ngrids+grid_id] = ax * ry * ry * rz * rz;
        gtox[13*ngrids+grid_id] = ax * ry * rz * rz * rz;
        gtox[14*ngrids+grid_id] = ax * rz * rz * rz * rz;
        gtoy[          grid_id] = ay * rx * rx * rx * rx;
        gtoy[1 *ngrids+grid_id] = ay * rx * rx * rx * ry +     bxxx;
        gtoy[2 *ngrids+grid_id] = ay * rx * rx * rx * rz;
        gtoy[3 *ngrids+grid_id] = ay * rx * rx * ry * ry + 2 * bxxy;
        gtoy[4 *ngrids+grid_id] = ay * rx * rx * ry * rz +     bxxz;
        gtoy[5 *ngrids+grid_id] = ay * rx * rx * rz * rz;
        gtoy[6 *ngrids+grid_id] = ay * rx * ry * ry * ry + 3 * bxyy;
        gtoy[7 *ngrids+grid_id] = ay * rx * ry * ry * rz + 2 * bxyz;
        gtoy[8 *ngrids+grid_id] = ay * rx * ry * rz * rz +     bxzz;
        gtoy[9 *ngrids+grid_id] = ay * rx * rz * rz * rz;
        gtoy[10*ngrids+grid_id] = ay * ry * ry * ry * ry + 4 * byyy;
        gtoy[11*ngrids+grid_id] = ay * ry * ry * ry * rz + 3 * byyz;
        gtoy[12*ngrids+grid_id] = ay * ry * ry * rz * rz + 2 * byzz;
        gtoy[13*ngrids+grid_id] = ay * ry * rz * rz * rz +     bzzz;
        gtoy[14*ngrids+grid_id] = ay * rz * rz * rz * rz;
        gtoz[          grid_id] = az * rx * rx * rx * rx;
        gtoz[1 *ngrids+grid_id] = az * rx * rx * rx * ry;
        gtoz[2 *ngrids+grid_id] = az * rx * rx * rx * rz +     bxxx;
        gtoz[3 *ngrids+grid_id] = az * rx * rx * ry * ry;
        gtoz[4 *ngrids+grid_id] = az * rx * rx * ry * rz +     bxxy;
        gtoz[5 *ngrids+grid_id] = az * rx * rx * rz * rz + 2 * bxxz;
        gtoz[6 *ngrids+grid_id] = az * rx * ry * ry * ry;
        gtoz[7 *ngrids+grid_id] = az * rx * ry * ry * rz +     bxyy;
        gtoz[8 *ngrids+grid_id] = az * rx * ry * rz * rz + 2 * bxyz;
        gtoz[9 *ngrids+grid_id] = az * rx * rz * rz * rz + 3 * bxzz;
        gtoz[10*ngrids+grid_id] = az * ry * ry * ry * ry;
        gtoz[11*ngrids+grid_id] = az * ry * ry * ry * rz +     byyy;
        gtoz[12*ngrids+grid_id] = az * ry * ry * rz * rz + 2 * byyz;
        gtoz[13*ngrids+grid_id] = az * ry * rz * rz * rz + 3 * byzz;
        gtoz[14*ngrids+grid_id] = az * rz * rz * rz * rz + 4 * bzzz;
    }
    else{
        double fx0[ANG+3], fy0[ANG+3], fz0[ANG+3];

        fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
        for (int lx = 1; lx <= ANG+2; lx++){
            fx0[lx] = fx0[lx-1] * rx;
            fy0[lx] = fy0[lx-1] * ry;
            fz0[lx] = fz0[lx-1] * rz;
        }

        _memset_cart<ANG>(gto+grid_id, 4, ngrids, nao);

        double fx1[ANG+1], fy1[ANG+1], fz1[ANG+1];
        for (int ip = 0; ip < offsets.nprim; ++ip) {
            const double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;

            _nabla1<ANG>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
            int i = 0;
            for (int lx = ANG; lx >= 0; lx--){
                for (int ly = ANG - lx; ly >= 0; ly--, i++){
                    int lz = ANG - lx - ly;
                    gto[  i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz0[lz];
                    gtox[ i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz0[lz];
                    gtoy[ i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz0[lz];
                    gtoz[ i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz1[lz];
                    //atomicAdd(gto +i*ngrids+grid_id, ce * fx0[lx] * fy0[ly] * fz0[lz]);
                    //atomicAdd(gtox+i*ngrids+grid_id, ce * fx1[lx] * fy0[ly] * fz0[lz]);
                    //atomicAdd(gtoy+i*ngrids+grid_id, ce * fx0[lx] * fy1[ly] * fz0[lz]);
                    //atomicAdd(gtoz+i*ngrids+grid_id, ce * fx0[lx] * fy0[ly] * fz1[lz]);
                }
            }
        }
    }
}

template <int ANG> __global__
static void _cart_kernel_deriv2(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto = offsets.data + i0 * ngrids;
    double* __restrict__ gtox = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz = offsets.data + (nao * 3 + i0) * ngrids;
    double* __restrict__ gtoxx = offsets.data + (nao * 4 + i0) * ngrids;
    double* __restrict__ gtoxy = offsets.data + (nao * 5 + i0) * ngrids;
    double* __restrict__ gtoxz = offsets.data + (nao * 6 + i0) * ngrids;
    double* __restrict__ gtoyy = offsets.data + (nao * 7 + i0) * ngrids;
    double* __restrict__ gtoyz = offsets.data + (nao * 8 + i0) * ngrids;
    double* __restrict__ gtozz = offsets.data + (nao * 9 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double fx0[ANG+3], fy0[ANG+3], fz0[ANG+3];
    double fx1[ANG+2], fy1[ANG+2], fz1[ANG+2];
    double fx2[ANG+1], fy2[ANG+1], fz2[ANG+1];

    fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
    for (int lx = 1; lx <= ANG+2; lx++){
        fx0[lx] = fx0[lx-1] * rx;
        fy0[lx] = fy0[lx-1] * ry;
        fz0[lx] = fz0[lx-1] * rz;
    }

    _memset_cart<ANG>(gto+grid_id, 10, ngrids, nao);

    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
        _nabla1<ANG+1>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
        _nabla1<ANG  >(fx2, fy2, fz2, fx1, fy1, fz1, exps[ip]);

        int i = 0;
        for (int lx = ANG; lx >= 0; lx--){
            for (int ly = ANG - lx; ly >= 0; ly--, i++){
                int lz = ANG - lx - ly;
                gto[  i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz0[lz];
                gtox[ i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz0[lz];
                gtoy[ i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz0[lz];
                gtoz[ i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz1[lz];
                gtoxx[i*ngrids + grid_id] += ce * fx2[lx] * fy0[ly] * fz0[lz];
                gtoxy[i*ngrids + grid_id] += ce * fx1[lx] * fy1[ly] * fz0[lz];
                gtoxz[i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz1[lz];
                gtoyy[i*ngrids + grid_id] += ce * fx0[lx] * fy2[ly] * fz0[lz];
                gtoyz[i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz1[lz];
                gtozz[i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz2[lz];
            }
        }
    }
}


template <int ANG> __global__
static void _cart_kernel_deriv3(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto    = offsets.data + i0 * ngrids;
    double* __restrict__ gtox   = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy   = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz   = offsets.data + (nao * 3 + i0) * ngrids;
    double* __restrict__ gtoxx  = offsets.data + (nao * 4 + i0) * ngrids;
    double* __restrict__ gtoxy  = offsets.data + (nao * 5 + i0) * ngrids;
    double* __restrict__ gtoxz  = offsets.data + (nao * 6 + i0) * ngrids;
    double* __restrict__ gtoyy  = offsets.data + (nao * 7 + i0) * ngrids;
    double* __restrict__ gtoyz  = offsets.data + (nao * 8 + i0) * ngrids;
    double* __restrict__ gtozz  = offsets.data + (nao * 9 + i0) * ngrids;
    double* __restrict__ gtoxxx = offsets.data + (nao * 10 + i0) * ngrids;
    double* __restrict__ gtoxxy = offsets.data + (nao * 11 + i0) * ngrids;
    double* __restrict__ gtoxxz = offsets.data + (nao * 12 + i0) * ngrids;
    double* __restrict__ gtoxyy = offsets.data + (nao * 13 + i0) * ngrids;
    double* __restrict__ gtoxyz = offsets.data + (nao * 14 + i0) * ngrids;
    double* __restrict__ gtoxzz = offsets.data + (nao * 15 + i0) * ngrids;
    double* __restrict__ gtoyyy = offsets.data + (nao * 16 + i0) * ngrids;
    double* __restrict__ gtoyyz = offsets.data + (nao * 17 + i0) * ngrids;
    double* __restrict__ gtoyzz = offsets.data + (nao * 18 + i0) * ngrids;
    double* __restrict__ gtozzz = offsets.data + (nao * 19 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double fx0[ANG+4], fy0[ANG+4], fz0[ANG+4];
    double fx1[ANG+3], fy1[ANG+3], fz1[ANG+3];
    double fx2[ANG+2], fy2[ANG+2], fz2[ANG+2];
    double fx3[ANG+1], fy3[ANG+1], fz3[ANG+1];

    fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
    for (int lx = 1; lx <= ANG+3; lx++){
        fx0[lx] = fx0[lx-1] * rx;
        fy0[lx] = fy0[lx-1] * ry;
        fz0[lx] = fz0[lx-1] * rz;
    }

    _memset_cart<ANG>(gto+grid_id, 20, ngrids, nao);

    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
        _nabla1<ANG+2>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
        _nabla1<ANG+1>(fx2, fy2, fz2, fx1, fy1, fz1, exps[ip]);
        _nabla1<ANG  >(fx3, fy3, fz3, fx2, fy2, fz2, exps[ip]);

        int i = 0;
        for (int lx = ANG; lx >= 0; lx--){
            for (int ly = ANG - lx; ly >= 0; ly--, i++){
                int lz = ANG - lx - ly;
                gto   [i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz0[lz];
                gtox  [i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz0[lz];
                gtoy  [i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz0[lz];
                gtoz  [i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz1[lz];
                gtoxx [i*ngrids + grid_id] += ce * fx2[lx] * fy0[ly] * fz0[lz];
                gtoxy [i*ngrids + grid_id] += ce * fx1[lx] * fy1[ly] * fz0[lz];
                gtoxz [i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz1[lz];
                gtoyy [i*ngrids + grid_id] += ce * fx0[lx] * fy2[ly] * fz0[lz];
                gtoyz [i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz1[lz];
                gtozz [i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz2[lz];
                gtoxxx[i*ngrids + grid_id] += ce * fx3[lx] * fy0[ly] * fz0[lz];
                gtoxxy[i*ngrids + grid_id] += ce * fx2[lx] * fy1[ly] * fz0[lz];
                gtoxxz[i*ngrids + grid_id] += ce * fx2[lx] * fy0[ly] * fz1[lz];
                gtoxyy[i*ngrids + grid_id] += ce * fx1[lx] * fy2[ly] * fz0[lz];
                gtoxyz[i*ngrids + grid_id] += ce * fx1[lx] * fy1[ly] * fz1[lz];
                gtoxzz[i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz2[lz];
                gtoyyy[i*ngrids + grid_id] += ce * fx0[lx] * fy3[ly] * fz0[lz];
                gtoyyz[i*ngrids + grid_id] += ce * fx0[lx] * fy2[ly] * fz1[lz];
                gtoyzz[i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz2[lz];
                gtozzz[i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz3[lz];
            }
        }
    }
}


template <int ANG> __global__
static void _cart_kernel_deriv4(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto     = offsets.data + i0 * ngrids;
    double* __restrict__ gtox    = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy    = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz    = offsets.data + (nao * 3 + i0) * ngrids;
    double* __restrict__ gtoxx   = offsets.data + (nao * 4 + i0) * ngrids;
    double* __restrict__ gtoxy   = offsets.data + (nao * 5 + i0) * ngrids;
    double* __restrict__ gtoxz   = offsets.data + (nao * 6 + i0) * ngrids;
    double* __restrict__ gtoyy   = offsets.data + (nao * 7 + i0) * ngrids;
    double* __restrict__ gtoyz   = offsets.data + (nao * 8 + i0) * ngrids;
    double* __restrict__ gtozz   = offsets.data + (nao * 9 + i0) * ngrids;
    double* __restrict__ gtoxxx  = offsets.data + (nao * 10 + i0) * ngrids;
    double* __restrict__ gtoxxy  = offsets.data + (nao * 11 + i0) * ngrids;
    double* __restrict__ gtoxxz  = offsets.data + (nao * 12 + i0) * ngrids;
    double* __restrict__ gtoxyy  = offsets.data + (nao * 13 + i0) * ngrids;
    double* __restrict__ gtoxyz  = offsets.data + (nao * 14 + i0) * ngrids;
    double* __restrict__ gtoxzz  = offsets.data + (nao * 15 + i0) * ngrids;
    double* __restrict__ gtoyyy  = offsets.data + (nao * 16 + i0) * ngrids;
    double* __restrict__ gtoyyz  = offsets.data + (nao * 17 + i0) * ngrids;
    double* __restrict__ gtoyzz  = offsets.data + (nao * 18 + i0) * ngrids;
    double* __restrict__ gtozzz  = offsets.data + (nao * 19 + i0) * ngrids;
    double* __restrict__ gtoxxxx = offsets.data + (nao * 20 + i0) * ngrids;
    double* __restrict__ gtoxxxy = offsets.data + (nao * 21 + i0) * ngrids;
    double* __restrict__ gtoxxxz = offsets.data + (nao * 22 + i0) * ngrids;
    double* __restrict__ gtoxxyy = offsets.data + (nao * 23 + i0) * ngrids;
    double* __restrict__ gtoxxyz = offsets.data + (nao * 24 + i0) * ngrids;
    double* __restrict__ gtoxxzz = offsets.data + (nao * 25 + i0) * ngrids;
    double* __restrict__ gtoxyyy = offsets.data + (nao * 26 + i0) * ngrids;
    double* __restrict__ gtoxyyz = offsets.data + (nao * 27 + i0) * ngrids;
    double* __restrict__ gtoxyzz = offsets.data + (nao * 28 + i0) * ngrids;
    double* __restrict__ gtoxzzz = offsets.data + (nao * 29 + i0) * ngrids;
    double* __restrict__ gtoyyyy = offsets.data + (nao * 30 + i0) * ngrids;
    double* __restrict__ gtoyyyz = offsets.data + (nao * 31 + i0) * ngrids;
    double* __restrict__ gtoyyzz = offsets.data + (nao * 32 + i0) * ngrids;
    double* __restrict__ gtoyzzz = offsets.data + (nao * 33 + i0) * ngrids;
    double* __restrict__ gtozzzz = offsets.data + (nao * 34 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double fx0[ANG+5], fy0[ANG+5], fz0[ANG+5];
    double fx1[ANG+4], fy1[ANG+4], fz1[ANG+4];
    double fx2[ANG+3], fy2[ANG+3], fz2[ANG+3];
    double fx3[ANG+2], fy3[ANG+2], fz3[ANG+2];
    double fx4[ANG+1], fy4[ANG+1], fz4[ANG+1];

    fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
    for (int lx = 1; lx <= ANG+4; lx++){
        fx0[lx] = fx0[lx-1] * rx;
        fy0[lx] = fy0[lx-1] * ry;
        fz0[lx] = fz0[lx-1] * rz;
    }

    _memset_cart<ANG>(gto+grid_id, 35, ngrids, nao);

    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
        _nabla1<ANG+3>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
        _nabla1<ANG+2>(fx2, fy2, fz2, fx1, fy1, fz1, exps[ip]);
        _nabla1<ANG+1>(fx3, fy3, fz3, fx2, fy2, fz2, exps[ip]);
        _nabla1<ANG  >(fx4, fy4, fz4, fx3, fy3, fz3, exps[ip]);
        int i = 0;
        for (int lx = ANG; lx >= 0; lx--){
            for (int ly = ANG - lx; ly >= 0; ly--, i++){
                int lz = ANG - lx - ly;
                gto    [i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz0[lz];
                gtox   [i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz0[lz];
                gtoy   [i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz0[lz];
                gtoz   [i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz1[lz];
                gtoxx  [i*ngrids + grid_id] += ce * fx2[lx] * fy0[ly] * fz0[lz];
                gtoxy  [i*ngrids + grid_id] += ce * fx1[lx] * fy1[ly] * fz0[lz];
                gtoxz  [i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz1[lz];
                gtoyy  [i*ngrids + grid_id] += ce * fx0[lx] * fy2[ly] * fz0[lz];
                gtoyz  [i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz1[lz];
                gtozz  [i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz2[lz];
                gtoxxx [i*ngrids + grid_id] += ce * fx3[lx] * fy0[ly] * fz0[lz];
                gtoxxy [i*ngrids + grid_id] += ce * fx2[lx] * fy1[ly] * fz0[lz];
                gtoxxz [i*ngrids + grid_id] += ce * fx2[lx] * fy0[ly] * fz1[lz];
                gtoxyy [i*ngrids + grid_id] += ce * fx1[lx] * fy2[ly] * fz0[lz];
                gtoxyz [i*ngrids + grid_id] += ce * fx1[lx] * fy1[ly] * fz1[lz];
                gtoxzz [i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz2[lz];
                gtoyyy [i*ngrids + grid_id] += ce * fx0[lx] * fy3[ly] * fz0[lz];
                gtoyyz [i*ngrids + grid_id] += ce * fx0[lx] * fy2[ly] * fz1[lz];
                gtoyzz [i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz2[lz];
                gtozzz [i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz3[lz];
                gtoxxxx[i*ngrids + grid_id] += ce * fx4[lx] * fy0[ly] * fz0[lz];
                gtoxxxy[i*ngrids + grid_id] += ce * fx3[lx] * fy1[ly] * fz0[lz];
                gtoxxxz[i*ngrids + grid_id] += ce * fx3[lx] * fy0[ly] * fz1[lz];
                gtoxxyy[i*ngrids + grid_id] += ce * fx2[lx] * fy2[ly] * fz0[lz];
                gtoxxyz[i*ngrids + grid_id] += ce * fx2[lx] * fy1[ly] * fz1[lz];
                gtoxxzz[i*ngrids + grid_id] += ce * fx2[lx] * fy0[ly] * fz2[lz];
                gtoxyyy[i*ngrids + grid_id] += ce * fx1[lx] * fy3[ly] * fz0[lz];
                gtoxyyz[i*ngrids + grid_id] += ce * fx1[lx] * fy2[ly] * fz1[lz];
                gtoxyzz[i*ngrids + grid_id] += ce * fx1[lx] * fy1[ly] * fz2[lz];
                gtoxzzz[i*ngrids + grid_id] += ce * fx1[lx] * fy0[ly] * fz3[lz];
                gtoyyyy[i*ngrids + grid_id] += ce * fx0[lx] * fy4[ly] * fz0[lz];
                gtoyyyz[i*ngrids + grid_id] += ce * fx0[lx] * fy3[ly] * fz1[lz];
                gtoyyzz[i*ngrids + grid_id] += ce * fx0[lx] * fy2[ly] * fz2[lz];
                gtoyzzz[i*ngrids + grid_id] += ce * fx0[lx] * fy1[ly] * fz3[lz];
                gtozzzz[i*ngrids + grid_id] += ce * fx0[lx] * fy0[ly] * fz4[lz];
            }
        }
    }
}

template <int ANG> __global__
static void _sph_kernel_deriv0(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }
    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto = offsets.data + i0 * ngrids;
    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double ce = 0;
    for (int ip = 0; ip < offsets.nprim; ++ip) {
        ce += coeffs[ip] * exp(-exps[ip] * rr);
    }
    ce *= offsets.fac;

    if (ANG == 2) {
        double g0 = ce * rx * rx;
        double g1 = ce * rx * ry;
        double g2 = ce * rx * rz;
        double g3 = ce * ry * ry;
        double g4 = ce * ry * rz;
        double g5 = ce * rz * rz;
        /*
        gto[         grid_id] = 1.092548430592079070 * g1;
        gto[1*ngrids+grid_id] = 1.092548430592079070 * g4;
        gto[2*ngrids+grid_id] = 0.630783130505040012 * g5 - 0.315391565252520002 * g0 - 0.315391565252520002 * g3;
        gto[3*ngrids+grid_id] = 1.092548430592079070 * g2;
        gto[4*ngrids+grid_id] = 0.546274215296039535 * g0 - 0.546274215296039535 * g3;
        */
        gto[         grid_id] = 1.092548430592079070 * g1;
        gto[1*ngrids+grid_id] = 1.092548430592079070 * g4;
        gto[2*ngrids+grid_id] = 0.630783130505040012 * g5 - 0.315391565252520002 * (g0 + g3);
        gto[3*ngrids+grid_id] = 1.092548430592079070 * g2;
        gto[4*ngrids+grid_id] = 0.546274215296039535 * (g0 - g3);
    } else if (ANG == 3) {
        double g0 = ce * rx * rx * rx;
        double g1 = ce * rx * rx * ry;
        double g2 = ce * rx * rx * rz;
        double g3 = ce * rx * ry * ry;
        double g4 = ce * rx * ry * rz;
        double g5 = ce * rx * rz * rz;
        double g6 = ce * ry * ry * ry;
        double g7 = ce * ry * ry * rz;
        double g8 = ce * ry * rz * rz;
        double g9 = ce * rz * rz * rz;
        /*
        gto[         grid_id] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
        gto[1*ngrids+grid_id] = 2.890611442640554055 * g4;
        gto[2*ngrids+grid_id] = 1.828183197857862944 * g8 - 0.457045799464465739 * g1 - 0.457045799464465739 * g6;
        gto[3*ngrids+grid_id] = 0.746352665180230782 * g9 - 1.119528997770346170 * g2 - 1.119528997770346170 * g7;
        gto[4*ngrids+grid_id] = 1.828183197857862944 * g5 - 0.457045799464465739 * g0 - 0.457045799464465739 * g3;
        gto[5*ngrids+grid_id] = 1.445305721320277020 * g2 - 1.445305721320277020 * g7;
        gto[6*ngrids+grid_id] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;
        */
        gto[         grid_id] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
        gto[1*ngrids+grid_id] = 2.890611442640554055 * g4;
        gto[2*ngrids+grid_id] = 1.828183197857862944 * g8 - 0.457045799464465739 * (g1 + g6);
        gto[3*ngrids+grid_id] = 0.746352665180230782 * g9 - 1.119528997770346170 * (g2 + g7);
        gto[4*ngrids+grid_id] = 1.828183197857862944 * g5 - 0.457045799464465739 * (g0 + g3);
        gto[5*ngrids+grid_id] = 1.445305721320277020 * (g2 - g7);
        gto[6*ngrids+grid_id] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;
    } else if (ANG == 4) {
        double g0  = ce * rx * rx * rx * rx;
        double g1  = ce * rx * rx * rx * ry;
        double g2  = ce * rx * rx * rx * rz;
        double g3  = ce * rx * rx * ry * ry;
        double g4  = ce * rx * rx * ry * rz;
        double g5  = ce * rx * rx * rz * rz;
        double g6  = ce * rx * ry * ry * ry;
        double g7  = ce * rx * ry * ry * rz;
        double g8  = ce * rx * ry * rz * rz;
        double g9  = ce * rx * rz * rz * rz;
        double g10 = ce * ry * ry * ry * ry;
        double g11 = ce * ry * ry * ry * rz;
        double g12 = ce * ry * ry * rz * rz;
        double g13 = ce * ry * rz * rz * rz;
        double g14 = ce * rz * rz * rz * rz;
        /*
        gto[         grid_id] = 2.503342941796704538 * g1 - 2.503342941796704530 * g6 ;
        gto[1*ngrids+grid_id] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
        gto[2*ngrids+grid_id] = 5.677048174545360108 * g8 - 0.946174695757560014 * g1 - 0.946174695757560014 * g6 ;
        gto[3*ngrids+grid_id] = 2.676186174229156671 * g13- 2.007139630671867500 * g4 - 2.007139630671867500 * g11;
        gto[4*ngrids+grid_id] = 0.317356640745612911 * g0 + 0.634713281491225822 * g3 - 2.538853125964903290 * g5 + 0.317356640745612911 * g10 - 2.538853125964903290 * g12 + 0.846284375321634430 * g14;
        gto[5*ngrids+grid_id] = 2.676186174229156671 * g9 - 2.007139630671867500 * g2 - 2.007139630671867500 * g7 ;
        gto[6*ngrids+grid_id] = 2.838524087272680054 * g5 + 0.473087347878780009 * g10- 0.473087347878780002 * g0 - 2.838524087272680050 * g12;
        gto[7*ngrids+grid_id] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
        gto[8*ngrids+grid_id] = 0.625835735449176134 * g0 - 3.755014412695056800 * g3 + 0.625835735449176134 * g10;
        */
        gto[         grid_id] = 2.503342941796704538 * (g1 - g6);
        gto[1*ngrids+grid_id] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
        gto[2*ngrids+grid_id] = 5.677048174545360108 * g8 - 0.946174695757560014 * (g1 + g6);
        gto[3*ngrids+grid_id] = 2.676186174229156671 * g13- 2.007139630671867500 * (g4 + g11);
        gto[4*ngrids+grid_id] = 0.317356640745612911 * (g0 + g10) + 0.634713281491225822 * g3 - 2.538853125964903290 * (g5 + g12) + 0.846284375321634430 * g14;
        gto[5*ngrids+grid_id] = 2.676186174229156671 * g9 - 2.007139630671867500 * (g2 + g7);
        gto[6*ngrids+grid_id] = 2.838524087272680054 * (g5 - g12) + 0.473087347878780009 * (g10 - g0);
        gto[7*ngrids+grid_id] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
        gto[8*ngrids+grid_id] = 0.625835735449176134 * (g0  + g10) - 3.755014412695056800 * g3;
    } else {
        double fx0[ANG+1], fy0[ANG+1], fz0[ANG+1];
        fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
        for (int lx = 1; lx <= ANG; lx++){
            fx0[lx] = fx0[lx-1] * rx;
            fy0[lx] = fy0[lx-1] * ry;
            fz0[lx] = fz0[lx-1] * rz;
        }

        _memset_sph<ANG>(gto+grid_id, 1, ngrids, 0);

        for (int ip = 0; ip < offsets.nprim; ++ip) {
            double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
            double g[(ANG+1)*(ANG+2)/2];
            _cart_gto<ANG>(g, ce, fx0, fy0, fz0); _cart2sph<ANG>(g, gto,   ngrids, grid_id);
        }
    }
}


template <int ANG> __global__
static void _sph_kernel_deriv1(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];

    double* __restrict__ gto = offsets.data + i0 * ngrids;
    double* __restrict__ gtox = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz = offsets.data + (nao * 3 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double ce = 0;
    double ce_2a = 0;
    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double c = coeffs[ip];
        double exp_ip = exps[ip];
        double e = exp(-exp_ip * rr);
        ce += c * e;
        ce_2a += c * e * exp_ip;
    }
    ce *= offsets.fac;
    ce_2a *= -2 * offsets.fac;

    if (ANG == 2) {
        double g0 = ce * rx * rx;
        double g1 = ce * rx * ry;
        double g2 = ce * rx * rz;
        double g3 = ce * ry * ry;
        double g4 = ce * ry * rz;
        double g5 = ce * rz * rz;
        /*
        gto[         grid_id] = 1.092548430592079070 * g1;
        gto[1*ngrids+grid_id] = 1.092548430592079070 * g4;
        gto[2*ngrids+grid_id] = 0.630783130505040012 * g5 - 0.315391565252520002 * g0 - 0.315391565252520002 * g3;
        gto[3*ngrids+grid_id] = 1.092548430592079070 * g2;
        gto[4*ngrids+grid_id] = 0.546274215296039535 * g0 - 0.546274215296039535 * g3;
        */
        gto[         grid_id] = 1.092548430592079070 * g1;
        gto[1*ngrids+grid_id] = 1.092548430592079070 * g4;
        gto[2*ngrids+grid_id] = 0.315391565252520002 * (2 * g5 - g0 - g3);
        gto[3*ngrids+grid_id] = 1.092548430592079070 * g2;
        gto[4*ngrids+grid_id] = 0.546274215296039535 * (g0 - g3);

        double ax = ce_2a * rx;
        double ax_ce  = ax * rx + ce;
        double ax_2ce = ax_ce  + ce;
        g0 = ax_2ce * rx;
        g1 = ax_ce  * ry;
        g2 = ax_ce  * rz;
        g3 = ax * ry * ry;
        g4 = ax * ry * rz;
        g5 = ax * rz * rz;
        gtox[         grid_id] = 1.092548430592079070 * g1;
        gtox[1*ngrids+grid_id] = 1.092548430592079070 * g4;
        gtox[2*ngrids+grid_id] = 0.315391565252520002 * (2 * g5 - g0 - g3);
        gtox[3*ngrids+grid_id] = 1.092548430592079070 * g2;
        gtox[4*ngrids+grid_id] = 0.546274215296039535 * (g0 - g3);

        double ay = ce_2a * ry;
        double ay_ce = ay * ry + ce;
        double ay_2ce = ay_ce + ce;
        g0 =            ay * rx * rx;
        g1 =              ay_ce * rx;
        g2 =            ay * rx * rz;
        g3 =             ay_2ce * ry;
        g4 =              ay_ce * rz;
        g5 =            ay * rz * rz;
        gtoy[         grid_id] = 1.092548430592079070 * g1;
        gtoy[1*ngrids+grid_id] = 1.092548430592079070 * g4;
        gtoy[2*ngrids+grid_id] = 0.315391565252520002 * (2 * g5 - g0 - g3);
        gtoy[3*ngrids+grid_id] = 1.092548430592079070 * g2;
        gtoy[4*ngrids+grid_id] = 0.546274215296039535 * (g0 - g3);

        double az = ce_2a * rz;
        double az_ce = az * rz + ce;
        double az_2ce = az_ce + ce;
        g0 = az * rx * rx;
        g1 = az * rx * ry;
        g2 = az_ce * rx;
        g3 = az * ry * ry;
        g4 = az_ce * ry;
        g5 = az_2ce * rz;
        gtoz[         grid_id] = 1.092548430592079070 * g1;
        gtoz[1*ngrids+grid_id] = 1.092548430592079070 * g4;
        gtoz[2*ngrids+grid_id] = 0.315391565252520002 * (2 * g5 - g0 - g3);
        gtoz[3*ngrids+grid_id] = 1.092548430592079070 * g2;
        gtoz[4*ngrids+grid_id] = 0.546274215296039535 * (g0 - g3);
    } else if (ANG == 3) {
        double g0 = ce * rx * rx * rx;
        double g1 = ce * rx * rx * ry;
        double g2 = ce * rx * rx * rz;
        double g3 = ce * rx * ry * ry;
        double g4 = ce * rx * ry * rz;
        double g5 = ce * rx * rz * rz;
        double g6 = ce * ry * ry * ry;
        double g7 = ce * ry * ry * rz;
        double g8 = ce * ry * rz * rz;
        double g9 = ce * rz * rz * rz;
        gto[         grid_id] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
        gto[1*ngrids+grid_id] = 2.890611442640554055 * g4;
        gto[2*ngrids+grid_id] = 1.828183197857862944 * g8 - 0.457045799464465739 * (g1 + g6);
        gto[3*ngrids+grid_id] = 0.746352665180230782 * g9 - 1.119528997770346170 * (g2 + g7);
        gto[4*ngrids+grid_id] = 1.828183197857862944 * g5 - 0.457045799464465739 * (g0 + g3);
        gto[5*ngrids+grid_id] = 1.445305721320277020 * (g2 - g7);
        gto[6*ngrids+grid_id] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;

        double ax = ce_2a * rx;
        double ax_ce = ax * rx + ce;
        double ax_2ce = ax_ce + ce;
        double ax_3ce = ax_2ce + ce;
        g0 = ax_3ce * rx * rx;
        g1 = ax_2ce * rx * ry;
        g2 = ax_2ce * rx * rz;
        g3 = ax_ce  * ry * ry;
        g4 = ax_ce  * ry * rz;
        g5 = ax_ce  * rz * rz;
        g6 = ax * ry * ry * ry;
        g7 = ax * ry * ry * rz;
        g8 = ax * ry * rz * rz;
        g9 = ax * rz * rz * rz;
        gtox[         grid_id] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
        gtox[1*ngrids+grid_id] = 2.890611442640554055 * g4;
        gtox[2*ngrids+grid_id] = 1.828183197857862944 * g8 - 0.457045799464465739 * (g1 + g6);
        gtox[3*ngrids+grid_id] = 0.746352665180230782 * g9 - 1.119528997770346170 * (g2 + g7);
        gtox[4*ngrids+grid_id] = 1.828183197857862944 * g5 - 0.457045799464465739 * (g0 + g3);
        gtox[5*ngrids+grid_id] = 1.445305721320277020 * (g2 - g7);
        gtox[6*ngrids+grid_id] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;

        double ay = ce_2a * ry;
        double ay_ce = ay * ry + ce;
        double ay_2ce = ay_ce + ce;
        double ay_3ce = ay_2ce + ce;
        g0 =   ay * rx * rx * rx;
        g1 =     ay_ce * rx * rx;
        g2 =   ay * rx * rx * rz;
        g3 =    ay_2ce * rx * ry;
        g4 =     ay_ce * rx * rz;
        g5 =   ay * rx * rz * rz;
        g6 =    ay_3ce * ry * ry;
        g7 =    ay_2ce * ry * rz;
        g8 =     ay_ce * rz * rz;
        g9 =   ay * rz * rz * rz;
        gtoy[         grid_id] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
        gtoy[1*ngrids+grid_id] = 2.890611442640554055 * g4;
        gtoy[2*ngrids+grid_id] = 1.828183197857862944 * g8 - 0.457045799464465739 * (g1 + g6);
        gtoy[3*ngrids+grid_id] = 0.746352665180230782 * g9 - 1.119528997770346170 * (g2 + g7);
        gtoy[4*ngrids+grid_id] = 1.828183197857862944 * g5 - 0.457045799464465739 * (g0 + g3);
        gtoy[5*ngrids+grid_id] = 1.445305721320277020 * (g2 - g7);
        gtoy[6*ngrids+grid_id] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;

        double az = ce_2a * rz;
        double az_ce = az * rz + ce;
        double az_2ce = az_ce + ce;
        double az_3ce = az_2ce + ce;
        g0 =  az * rx * rx * rx;
        g1 =  az * rx * rx * ry;
        g2 =    az_ce * rx * rx;
        g3 =  az * rx * ry * ry;
        g4 =    az_ce * rx * ry;
        g5 =   az_2ce * rx * rz;
        g6 =  az * ry * ry * ry;
        g7 =    az_ce * ry * ry;
        g8 =   az_2ce * ry * rz;
        g9 =   az_3ce * rz * rz;
        gtoz[         grid_id] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
        gtoz[1*ngrids+grid_id] = 2.890611442640554055 * g4;
        gtoz[2*ngrids+grid_id] = 1.828183197857862944 * g8 - 0.457045799464465739 * (g1 + g6);
        gtoz[3*ngrids+grid_id] = 0.746352665180230782 * g9 - 1.119528997770346170 * (g2 + g7);
        gtoz[4*ngrids+grid_id] = 1.828183197857862944 * g5 - 0.457045799464465739 * (g0 + g3);
        gtoz[5*ngrids+grid_id] = 1.445305721320277020 * (g2 - g7);
        gtoz[6*ngrids+grid_id] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;
    } else if (ANG == 4) {
        double g0  = ce * rx * rx * rx * rx;
        double g1  = ce * rx * rx * rx * ry;
        double g2  = ce * rx * rx * rx * rz;
        double g3  = ce * rx * rx * ry * ry;
        double g4  = ce * rx * rx * ry * rz;
        double g5  = ce * rx * rx * rz * rz;
        double g6  = ce * rx * ry * ry * ry;
        double g7  = ce * rx * ry * ry * rz;
        double g8  = ce * rx * ry * rz * rz;
        double g9  = ce * rx * rz * rz * rz;
        double g10 = ce * ry * ry * ry * ry;
        double g11 = ce * ry * ry * ry * rz;
        double g12 = ce * ry * ry * rz * rz;
        double g13 = ce * ry * rz * rz * rz;
        double g14 = ce * rz * rz * rz * rz;
        gto[          grid_id] = 2.503342941796704538 * (g1 - g6);
        gto[1 *ngrids+grid_id] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
        gto[2 *ngrids+grid_id] = 5.677048174545360108 * g8 - 0.946174695757560014 * (g1 + g6);
        gto[3 *ngrids+grid_id] = 2.676186174229156671 * g13 - 2.007139630671867500 * (g4 + g11);
        gto[4 *ngrids+grid_id] = 0.317356640745612911 * (g0 + g10) + 0.634713281491225822 * g3 - 2.538853125964903290 * (g5 + g12) + 0.846284375321634430 * g14;
        gto[5 *ngrids+grid_id] = 2.676186174229156671 * g9 - 2.007139630671867500 * (g2 + g7);
        gto[6 *ngrids+grid_id] = 2.838524087272680054 * (g5 - g12) + 0.473087347878780009 * (g10 - g0);
        gto[7 *ngrids+grid_id] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
        gto[8 *ngrids+grid_id] = 0.625835735449176134 * (g0 + g10) - 3.755014412695056800 * g3;

        double ax = ce_2a * rx;
        g0  = (ax * rx + 4 * ce) * rx * rx * rx;
        g1  = (ax * rx + 3 * ce) * rx * rx * ry;
        g2  = (ax * rx + 3 * ce) * rx * rx * rz;
        g3  = (ax * rx + 2 * ce) * rx * ry * ry;
        g4  = (ax * rx + 2 * ce) * rx * ry * rz;
        g5  = (ax * rx + 2 * ce) * rx * rz * rz;
        g6  = (ax * rx +     ce) * ry * ry * ry;
        g7  = (ax * rx +     ce) * ry * ry * rz;
        g8  = (ax * rx +     ce) * ry * rz * rz;
        g9  = (ax * rx +     ce) * rz * rz * rz;
        g10 = ax * ry * ry * ry * ry;
        g11 = ax * ry * ry * ry * rz;
        g12 = ax * ry * ry * rz * rz;
        g13 = ax * ry * rz * rz * rz;
        g14 = ax * rz * rz * rz * rz;
        gtox[          grid_id] = 2.503342941796704538 * (g1 - g6) ;
        gtox[1 *ngrids+grid_id] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
        gtox[2 *ngrids+grid_id] = 5.677048174545360108 * g8 - 0.946174695757560014 * (g1 + g6);
        gtox[3 *ngrids+grid_id] = 2.676186174229156671 * g13 - 2.007139630671867500 * (g4 + g11);
        gtox[4 *ngrids+grid_id] = 0.317356640745612911 * (g0 + g10) + 0.634713281491225822 * g3 - 2.538853125964903290 * (g5 + g12) + 0.846284375321634430 * g14;
        gtox[5 *ngrids+grid_id] = 2.676186174229156671 * g9 - 2.007139630671867500 * (g2 + g7);
        gtox[6 *ngrids+grid_id] = 2.838524087272680054 * (g5 - g12) + 0.473087347878780009 * (g10 - g0);
        gtox[7 *ngrids+grid_id] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
        gtox[8 *ngrids+grid_id] = 0.625835735449176134 * (g0 + g10) - 3.755014412695056800 * g3;

        double ay = ce_2a * ry;
        g0  = ay * rx * rx * rx * rx;
        g1  = (ay * ry +     ce) * rx * rx * rx;
        g2  = ay * rx * rx * rx * rz;
        g3  = (ay * ry + 2 * ce) * rx * rx * ry;
        g4  = (ay * ry +     ce) * rx * rx * rz;
        g5  = ay * rx * rx * rz * rz;
        g6  = (ay * ry + 3 * ce) * rx * ry * ry;
        g7  = (ay * ry + 2 * ce) * rx * ry * rz;
        g8  = (ay * ry +     ce) * rx * rz * rz;
        g9  = ay * rx * rz * rz * rz;
        g10 = (ay * ry + 4 * ce) * ry * ry * ry;
        g11 = (ay * ry + 3 * ce) * ry * ry * rz;
        g12 = (ay * ry + 2 * ce) * ry * rz * rz;
        g13 = (ay * ry +     ce) * rz * rz * rz;
        g14 = ay * rz * rz * rz * rz;
        gtoy[          grid_id] = 2.503342941796704538 * (g1 - g6) ;
        gtoy[1 *ngrids+grid_id] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
        gtoy[2 *ngrids+grid_id] = 5.677048174545360108 * g8 - 0.946174695757560014 * (g1 + g6);
        gtoy[3 *ngrids+grid_id] = 2.676186174229156671 * g13 - 2.007139630671867500 * (g4 + g11);
        gtoy[4 *ngrids+grid_id] = 0.317356640745612911 * (g0 + g10) + 0.634713281491225822 * g3 - 2.538853125964903290 * (g5 + g12) + 0.846284375321634430 * g14;
        gtoy[5 *ngrids+grid_id] = 2.676186174229156671 * g9 - 2.007139630671867500 * (g2 + g7);
        gtoy[6 *ngrids+grid_id] = 2.838524087272680054 * (g5 - g12) + 0.473087347878780009 * (g10 - g0);
        gtoy[7 *ngrids+grid_id] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
        gtoy[8 *ngrids+grid_id] = 0.625835735449176134 * (g0 + g10) - 3.755014412695056800 * g3;

        double az = ce_2a * rz;
        g0  = az * rx * rx * rx * rx;
        g1  = az * rx * rx * rx * ry;
        g2  = (az * rz +     ce) * rx * rx * rx;
        g3  = az * rx * rx * ry * ry;
        g4  = (az * rz +     ce) * rx * rx * ry;
        g5  = (az * rz + 2 * ce) * rx * rx * rz;
        g6  = az * rx * ry * ry * ry;
        g7  = (az * rz +     ce) * rx * ry * ry;
        g8  = (az * rz + 2 * ce) * rx * ry * rz;
        g9  = (az * rz + 3 * ce) * rx * rz * rz;
        g10 = az * ry * ry * ry * ry;
        g11 = (az * rz +     ce) * ry * ry * ry;
        g12 = (az * rz + 2 * ce) * ry * ry * rz;
        g13 = (az * rz + 3 * ce) * ry * rz * rz;
        g14 = (az * rz + 4 * ce) * rz * rz * rz;
        gtoz[          grid_id] = 2.503342941796704538 * (g1 - g6) ;
        gtoz[1 *ngrids+grid_id] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
        gtoz[2 *ngrids+grid_id] = 5.677048174545360108 * g8 - 0.946174695757560014 * (g1 + g6);
        gtoz[3 *ngrids+grid_id] = 2.676186174229156671 * g13 - 2.007139630671867500 * (g4 + g11);
        gtoz[4 *ngrids+grid_id] = 0.317356640745612911 * (g0 + g10) + 0.634713281491225822 * g3 - 2.538853125964903290 * (g5 + g12) + 0.846284375321634430 * g14;
        gtoz[5 *ngrids+grid_id] = 2.676186174229156671 * g9 - 2.007139630671867500 * (g2 + g7);
        gtoz[6 *ngrids+grid_id] = 2.838524087272680054 * (g5 - g12) + 0.473087347878780009 * (g10 - g0);
        gtoz[7 *ngrids+grid_id] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
        gtoz[8 *ngrids+grid_id] = 0.625835735449176134 * (g0 + g10) - 3.755014412695056800 * g3;
    } else {
        double fx0[ANG+2], fy0[ANG+2], fz0[ANG+2];
        fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
        for (int lx = 1; lx <= ANG+1; lx++){
            fx0[lx] = fx0[lx-1] * rx;
            fy0[lx] = fy0[lx-1] * ry;
            fz0[lx] = fz0[lx-1] * rz;
        }
        double fx1[ANG+1], fy1[ANG+1], fz1[ANG+1];
        
        _memset_sph<ANG>(gto+grid_id, 4, ngrids, nao);

        for (int ip = 0; ip < offsets.nprim; ++ip) {
            double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
            _nabla1<ANG>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
            double g[(ANG+1)*(ANG+2)/2];
            _cart_gto<ANG>(g, ce, fx0, fy0, fz0); _cart2sph<ANG>(g, gto,   ngrids, grid_id);
            _cart_gto<ANG>(g, ce, fx1, fy0, fz0); _cart2sph<ANG>(g, gtox,  ngrids, grid_id);
            _cart_gto<ANG>(g, ce, fx0, fy1, fz0); _cart2sph<ANG>(g, gtoy,  ngrids, grid_id);
            _cart_gto<ANG>(g, ce, fx0, fy0, fz1); _cart2sph<ANG>(g, gtoz,  ngrids, grid_id);
        }
    }
}

template <int ANG> __global__
static void _sph_kernel_deriv2(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto = offsets.data + i0 * ngrids;
    double* __restrict__ gtox = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz = offsets.data + (nao * 3 + i0) * ngrids;
    double* __restrict__ gtoxx = offsets.data + (nao * 4 + i0) * ngrids;
    double* __restrict__ gtoxy = offsets.data + (nao * 5 + i0) * ngrids;
    double* __restrict__ gtoxz = offsets.data + (nao * 6 + i0) * ngrids;
    double* __restrict__ gtoyy = offsets.data + (nao * 7 + i0) * ngrids;
    double* __restrict__ gtoyz = offsets.data + (nao * 8 + i0) * ngrids;
    double* __restrict__ gtozz = offsets.data + (nao * 9 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double fx0[ANG+3], fy0[ANG+3], fz0[ANG+3];
    fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
#pragma unroll
    for (int lx = 1; lx <= ANG+2; lx++){
        fx0[lx] = fx0[lx-1] * rx;
        fy0[lx] = fy0[lx-1] * ry;
        fz0[lx] = fz0[lx-1] * rz;
    }
    double fx1[ANG+2], fy1[ANG+2], fz1[ANG+2];
    double fx2[ANG+1], fy2[ANG+1], fz2[ANG+1];

    _memset_sph<ANG>(gto+grid_id, 10, ngrids, nao);

    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
        _nabla1<ANG+1>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
        _nabla1<ANG  >(fx2, fy2, fz2, fx1, fy1, fz1, exps[ip]);

        double g[(ANG+1)*(ANG+2)/2];
        _cart_gto<ANG>(g, ce, fx0, fy0, fz0); _cart2sph<ANG>(g, gto,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz0); _cart2sph<ANG>(g, gtox,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz0); _cart2sph<ANG>(g, gtoy,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz1); _cart2sph<ANG>(g, gtoz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy0, fz0); _cart2sph<ANG>(g, gtoxx, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy1, fz0); _cart2sph<ANG>(g, gtoxy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz1); _cart2sph<ANG>(g, gtoxz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy2, fz0); _cart2sph<ANG>(g, gtoyy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz1); _cart2sph<ANG>(g, gtoyz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz2); _cart2sph<ANG>(g, gtozz, ngrids, grid_id);
    }
}


template <int ANG> __global__
static void _sph_kernel_deriv3(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto    = offsets.data + i0 * ngrids;
    double* __restrict__ gtox   = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy   = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz   = offsets.data + (nao * 3 + i0) * ngrids;
    double* __restrict__ gtoxx  = offsets.data + (nao * 4 + i0) * ngrids;
    double* __restrict__ gtoxy  = offsets.data + (nao * 5 + i0) * ngrids;
    double* __restrict__ gtoxz  = offsets.data + (nao * 6 + i0) * ngrids;
    double* __restrict__ gtoyy  = offsets.data + (nao * 7 + i0) * ngrids;
    double* __restrict__ gtoyz  = offsets.data + (nao * 8 + i0) * ngrids;
    double* __restrict__ gtozz  = offsets.data + (nao * 9 + i0) * ngrids;
    double* __restrict__ gtoxxx = offsets.data + (nao * 10 + i0) * ngrids;
    double* __restrict__ gtoxxy = offsets.data + (nao * 11 + i0) * ngrids;
    double* __restrict__ gtoxxz = offsets.data + (nao * 12 + i0) * ngrids;
    double* __restrict__ gtoxyy = offsets.data + (nao * 13 + i0) * ngrids;
    double* __restrict__ gtoxyz = offsets.data + (nao * 14 + i0) * ngrids;
    double* __restrict__ gtoxzz = offsets.data + (nao * 15 + i0) * ngrids;
    double* __restrict__ gtoyyy = offsets.data + (nao * 16 + i0) * ngrids;
    double* __restrict__ gtoyyz = offsets.data + (nao * 17 + i0) * ngrids;
    double* __restrict__ gtoyzz = offsets.data + (nao * 18 + i0) * ngrids;
    double* __restrict__ gtozzz = offsets.data + (nao * 19 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double fx0[ANG+4], fy0[ANG+4], fz0[ANG+4];
    fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
#pragma unroll
    for (int lx = 1; lx <= ANG+3; lx++){
        fx0[lx] = fx0[lx-1] * rx;
        fy0[lx] = fy0[lx-1] * ry;
        fz0[lx] = fz0[lx-1] * rz;
    }
    double fx1[ANG+3], fy1[ANG+3], fz1[ANG+3];
    double fx2[ANG+2], fy2[ANG+2], fz2[ANG+2];
    double fx3[ANG+1], fy3[ANG+1], fz3[ANG+1];

    _memset_sph<ANG>(gto+grid_id, 20, ngrids, nao);

    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
        _nabla1<ANG+2>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
        _nabla1<ANG+1>(fx2, fy2, fz2, fx1, fy1, fz1, exps[ip]);
        _nabla1<ANG  >(fx3, fy3, fz3, fx2, fy2, fz2, exps[ip]);

        double g[(ANG+1)*(ANG+2)/2];
        _cart_gto<ANG>(g, ce, fx0, fy0, fz0); _cart2sph<ANG>(g, gto,    ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz0); _cart2sph<ANG>(g, gtox,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz0); _cart2sph<ANG>(g, gtoy,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz1); _cart2sph<ANG>(g, gtoz,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy0, fz0); _cart2sph<ANG>(g, gtoxx,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy1, fz0); _cart2sph<ANG>(g, gtoxy,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz1); _cart2sph<ANG>(g, gtoxz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy2, fz0); _cart2sph<ANG>(g, gtoyy,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz1); _cart2sph<ANG>(g, gtoyz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz2); _cart2sph<ANG>(g, gtozz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx3, fy0, fz0); _cart2sph<ANG>(g, gtoxxx, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy1, fz0); _cart2sph<ANG>(g, gtoxxy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy0, fz1); _cart2sph<ANG>(g, gtoxxz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy2, fz0); _cart2sph<ANG>(g, gtoxyy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy1, fz1); _cart2sph<ANG>(g, gtoxyz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz2); _cart2sph<ANG>(g, gtoxzz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy3, fz0); _cart2sph<ANG>(g, gtoyyy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy2, fz1); _cart2sph<ANG>(g, gtoyyz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz2); _cart2sph<ANG>(g, gtoyzz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz3); _cart2sph<ANG>(g, gtozzz, ngrids, grid_id);
    }
}


template <int ANG> __global__
static void _sph_kernel_deriv4(BasOffsets offsets, GTOValEnvVars gto_envs)
{
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (grid_id >= ngrids) {
        return;
    }

    int bas_id = blockIdx.y;
    int natm = gto_envs.natm;
    int nao = offsets.nao;
    int local_ish = offsets.bas_off + bas_id;
    int glob_ish = offsets.bas_indices[local_ish];
    int atm_id = gto_envs.bas_atom[glob_ish];
    size_t i0 = offsets.ao_loc[local_ish];
    double* __restrict__ gto     = offsets.data + i0 * ngrids;
    double* __restrict__ gtox    = offsets.data + (nao * 1 + i0) * ngrids;
    double* __restrict__ gtoy    = offsets.data + (nao * 2 + i0) * ngrids;
    double* __restrict__ gtoz    = offsets.data + (nao * 3 + i0) * ngrids;
    double* __restrict__ gtoxx   = offsets.data + (nao * 4 + i0) * ngrids;
    double* __restrict__ gtoxy   = offsets.data + (nao * 5 + i0) * ngrids;
    double* __restrict__ gtoxz   = offsets.data + (nao * 6 + i0) * ngrids;
    double* __restrict__ gtoyy   = offsets.data + (nao * 7 + i0) * ngrids;
    double* __restrict__ gtoyz   = offsets.data + (nao * 8 + i0) * ngrids;
    double* __restrict__ gtozz   = offsets.data + (nao * 9 + i0) * ngrids;
    double* __restrict__ gtoxxx  = offsets.data + (nao * 10 + i0) * ngrids;
    double* __restrict__ gtoxxy  = offsets.data + (nao * 11 + i0) * ngrids;
    double* __restrict__ gtoxxz  = offsets.data + (nao * 12 + i0) * ngrids;
    double* __restrict__ gtoxyy  = offsets.data + (nao * 13 + i0) * ngrids;
    double* __restrict__ gtoxyz  = offsets.data + (nao * 14 + i0) * ngrids;
    double* __restrict__ gtoxzz  = offsets.data + (nao * 15 + i0) * ngrids;
    double* __restrict__ gtoyyy  = offsets.data + (nao * 16 + i0) * ngrids;
    double* __restrict__ gtoyyz  = offsets.data + (nao * 17 + i0) * ngrids;
    double* __restrict__ gtoyzz  = offsets.data + (nao * 18 + i0) * ngrids;
    double* __restrict__ gtozzz  = offsets.data + (nao * 19 + i0) * ngrids;
    double* __restrict__ gtoxxxx = offsets.data + (nao * 20 + i0) * ngrids;
    double* __restrict__ gtoxxxy = offsets.data + (nao * 21 + i0) * ngrids;
    double* __restrict__ gtoxxxz = offsets.data + (nao * 22 + i0) * ngrids;
    double* __restrict__ gtoxxyy = offsets.data + (nao * 23 + i0) * ngrids;
    double* __restrict__ gtoxxyz = offsets.data + (nao * 24 + i0) * ngrids;
    double* __restrict__ gtoxxzz = offsets.data + (nao * 25 + i0) * ngrids;
    double* __restrict__ gtoxyyy = offsets.data + (nao * 26 + i0) * ngrids;
    double* __restrict__ gtoxyyz = offsets.data + (nao * 27 + i0) * ngrids;
    double* __restrict__ gtoxyzz = offsets.data + (nao * 28 + i0) * ngrids;
    double* __restrict__ gtoxzzz = offsets.data + (nao * 29 + i0) * ngrids;
    double* __restrict__ gtoyyyy = offsets.data + (nao * 30 + i0) * ngrids;
    double* __restrict__ gtoyyyz = offsets.data + (nao * 31 + i0) * ngrids;
    double* __restrict__ gtoyyzz = offsets.data + (nao * 32 + i0) * ngrids;
    double* __restrict__ gtoyzzz = offsets.data + (nao * 33 + i0) * ngrids;
    double* __restrict__ gtozzzz = offsets.data + (nao * 34 + i0) * ngrids;

    double *atom_coordx = gto_envs.atom_coordx;
    double *atom_coordy = gto_envs.atom_coordx + natm;
    double *atom_coordz = gto_envs.atom_coordx + natm * 2;
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double rx = gridx[grid_id] - atom_coordx[atm_id];
    double ry = gridy[grid_id] - atom_coordy[atm_id];
    double rz = gridz[grid_id] - atom_coordz[atm_id];
    double rr = rx * rx + ry * ry + rz * rz;
    double *exps = gto_envs.env + gto_envs.bas_exp[glob_ish];
    double *coeffs = gto_envs.env + gto_envs.bas_coeff[glob_ish];

    double fx0[ANG+5], fy0[ANG+5], fz0[ANG+5];
    fx0[0] = 1.0; fy0[0] = 1.0; fz0[0] = 1.0;
#pragma unroll
    for (int lx = 1; lx <= ANG+4; lx++){
        fx0[lx] = fx0[lx-1] * rx;
        fy0[lx] = fy0[lx-1] * ry;
        fz0[lx] = fz0[lx-1] * rz;
    }
    double fx1[ANG+4], fy1[ANG+4], fz1[ANG+4];
    double fx2[ANG+3], fy2[ANG+3], fz2[ANG+3];
    double fx3[ANG+2], fy3[ANG+2], fz3[ANG+2];
    double fx4[ANG+1], fy4[ANG+1], fz4[ANG+1];

    _memset_sph<ANG>(gto+grid_id, 35, ngrids, nao);

    for (int ip = 0; ip < offsets.nprim; ++ip) {
        double ce = coeffs[ip] * exp(-exps[ip] * rr) * offsets.fac;
        _nabla1<ANG+3>(fx1, fy1, fz1, fx0, fy0, fz0, exps[ip]);
        _nabla1<ANG+2>(fx2, fy2, fz2, fx1, fy1, fz1, exps[ip]);
        _nabla1<ANG+1>(fx3, fy3, fz3, fx2, fy2, fz2, exps[ip]);
        _nabla1<ANG  >(fx4, fy4, fz4, fx3, fy3, fz3, exps[ip]);

        double g[(ANG+1)*(ANG+2)/2];
        _cart_gto<ANG>(g, ce, fx0, fy0, fz0); _cart2sph<ANG>(g, gto,     ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz0); _cart2sph<ANG>(g, gtox,    ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz0); _cart2sph<ANG>(g, gtoy,    ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz1); _cart2sph<ANG>(g, gtoz,    ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy0, fz0); _cart2sph<ANG>(g, gtoxx,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy1, fz0); _cart2sph<ANG>(g, gtoxy,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz1); _cart2sph<ANG>(g, gtoxz,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy2, fz0); _cart2sph<ANG>(g, gtoyy,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz1); _cart2sph<ANG>(g, gtoyz,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz2); _cart2sph<ANG>(g, gtozz,   ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx3, fy0, fz0); _cart2sph<ANG>(g, gtoxxx,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy1, fz0); _cart2sph<ANG>(g, gtoxxy,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy0, fz1); _cart2sph<ANG>(g, gtoxxz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy2, fz0); _cart2sph<ANG>(g, gtoxyy,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy1, fz1); _cart2sph<ANG>(g, gtoxyz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz2); _cart2sph<ANG>(g, gtoxzz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy3, fz0); _cart2sph<ANG>(g, gtoyyy,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy2, fz1); _cart2sph<ANG>(g, gtoyyz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz2); _cart2sph<ANG>(g, gtoyzz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz3); _cart2sph<ANG>(g, gtozzz,  ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx4, fy0, fz0); _cart2sph<ANG>(g, gtoxxxx, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx3, fy1, fz0); _cart2sph<ANG>(g, gtoxxxy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx3, fy0, fz1); _cart2sph<ANG>(g, gtoxxxz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy2, fz0); _cart2sph<ANG>(g, gtoxxyy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy1, fz1); _cart2sph<ANG>(g, gtoxxyz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx2, fy0, fz2); _cart2sph<ANG>(g, gtoxxzz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy3, fz0); _cart2sph<ANG>(g, gtoxyyy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy2, fz1); _cart2sph<ANG>(g, gtoxyyz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy1, fz2); _cart2sph<ANG>(g, gtoxyzz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx1, fy0, fz3); _cart2sph<ANG>(g, gtoxzzz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy4, fz0); _cart2sph<ANG>(g, gtoyyyy, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy3, fz1); _cart2sph<ANG>(g, gtoyyyz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy2, fz2); _cart2sph<ANG>(g, gtoyyzz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy1, fz3); _cart2sph<ANG>(g, gtoyzzz, ngrids, grid_id);
        _cart_gto<ANG>(g, ce, fx0, fy0, fz4); _cart2sph<ANG>(g, gtozzzz, ngrids, grid_id);
    }
}

extern "C" {
inline double CINTcommon_fac_sp(int l)
{
        switch (l) {
                case 0: return 0.282094791773878143;
                case 1: return 0.488602511902919921;
                default: return 1;
        }
}

int GDFTeval_gto(cudaStream_t stream, double *ao, int deriv, int cart,
                 double *grids, int ngrids,
                 int *bas_indices,
                 int *ao_loc, int nao,
                 int *ctr_offsets, int nctr,
                 int *local_ctr_offsets,
                 int *bas, GTOValEnvVars *gto_envs)
{
    BasOffsets offsets;
    //DEVICE_INIT(double, d_grids, grids, ngrids * 3);
    offsets.gridx = grids;//d_grids;
    offsets.ngrids = ngrids;
    offsets.data = ao;
    offsets.ao_loc = ao_loc;
    offsets.bas_indices = bas_indices;
    offsets.nbas = local_ctr_offsets[nctr];
    offsets.nao = nao;
    dim3 threads(NG_PER_BLOCK);
    dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);

    for (int ictr = 0; ictr < nctr; ++ictr) {
        int local_ish = local_ctr_offsets[ictr];
        int glob_ish = ctr_offsets[ictr]; //bas_indices[local_ish];
        int l = bas[ANG_OF+glob_ish*BAS_SLOTS];
        offsets.bas_off = local_ish;
        offsets.nprim = bas[NPRIM_OF+glob_ish*BAS_SLOTS];
        offsets.fac = CINTcommon_fac_sp(l);
        blocks.y = local_ctr_offsets[ictr+1] - local_ctr_offsets[ictr];
        if (blocks.y == 0){
            continue;
        }
        switch (deriv) {
        case 0:
            if (cart == 1) {
                switch (l) {
                case 0: _cart_kernel_deriv0<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv0<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _cart_kernel_deriv0<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _cart_kernel_deriv0<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _cart_kernel_deriv0<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _cart_kernel_deriv0<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _cart_kernel_deriv0<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _cart_kernel_deriv0<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _cart_kernel_deriv0<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default:fprintf(stderr, "l = %d not supported\n", l); }
            } else {
                switch (l) {
                case 0: _cart_kernel_deriv0<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv0<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _sph_kernel_deriv0 <2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _sph_kernel_deriv0 <3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _sph_kernel_deriv0 <4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _sph_kernel_deriv0 <5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _sph_kernel_deriv0 <6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _sph_kernel_deriv0 <7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _sph_kernel_deriv0 <8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); }
            }
            break;
        case 1:
            if (cart == 1) {
                switch (l) {
                case 0: _cart_kernel_deriv1<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv1<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _cart_kernel_deriv1<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _cart_kernel_deriv1<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _cart_kernel_deriv1<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _cart_kernel_deriv1<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _cart_kernel_deriv1<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _cart_kernel_deriv1<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _cart_kernel_deriv1<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); }
            } else {
                switch (l) {
                case 0: _cart_kernel_deriv1<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv1<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _sph_kernel_deriv1 <2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _sph_kernel_deriv1 <3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _sph_kernel_deriv1 <4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _sph_kernel_deriv1 <5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _sph_kernel_deriv1 <6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _sph_kernel_deriv1 <7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _sph_kernel_deriv1 <8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); }
            }
            break;
        case 2:
            if (cart == 1){
                switch (l) {
                case 0: _cart_kernel_deriv2<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv2<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _cart_kernel_deriv2<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _cart_kernel_deriv2<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _cart_kernel_deriv2<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _cart_kernel_deriv2<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _cart_kernel_deriv2<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _cart_kernel_deriv2<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _cart_kernel_deriv2<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); break;}
            } else {
                switch(l){
                case 0: _cart_kernel_deriv2<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv2<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _sph_kernel_deriv2<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _sph_kernel_deriv2<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _sph_kernel_deriv2<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _sph_kernel_deriv2<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _sph_kernel_deriv2<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _sph_kernel_deriv2<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _sph_kernel_deriv2<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); break; }
                }
            break;
        case 3:
            if (cart == 1){
                switch (l) {
                case 0: _cart_kernel_deriv3<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv3<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _cart_kernel_deriv3<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _cart_kernel_deriv3<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _cart_kernel_deriv3<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _cart_kernel_deriv3<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _cart_kernel_deriv3<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _cart_kernel_deriv3<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _cart_kernel_deriv3<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); break; }
            } else {
                switch(l){
                case 0: _cart_kernel_deriv3<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv3<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _sph_kernel_deriv3<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _sph_kernel_deriv3<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _sph_kernel_deriv3<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _sph_kernel_deriv3<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _sph_kernel_deriv3<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _sph_kernel_deriv3<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _sph_kernel_deriv3<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); break; }
                }
            break;
        case 4:
            if (cart == 1){
                switch (l) {
                case 0: _cart_kernel_deriv4<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv4<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _cart_kernel_deriv4<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _cart_kernel_deriv4<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _cart_kernel_deriv4<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _cart_kernel_deriv4<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _cart_kernel_deriv4<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _cart_kernel_deriv4<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _cart_kernel_deriv4<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); break; }
            } else {
                switch(l){
                case 0: _cart_kernel_deriv4<0> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 1: _cart_kernel_deriv4<1> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 2: _sph_kernel_deriv4<2> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 3: _sph_kernel_deriv4<3> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 4: _sph_kernel_deriv4<4> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 5: _sph_kernel_deriv4<5> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 6: _sph_kernel_deriv4<6> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 7: _sph_kernel_deriv4<7> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                case 8: _sph_kernel_deriv4<8> <<<blocks, threads, 0, stream>>>(offsets, *gto_envs); break;
                default: fprintf(stderr, "l = %d not supported\n", l); break; }
            }
            break;
        default:
            fprintf(stderr, "deriv %d not supported\n", deriv);
            return 1;
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error of GDFTeval_gto_kernel: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    //FREE(d_grids);
    return 0;
}

int GDFTscreen_index(cudaStream_t stream, int *non0shl_idx, double cutoff,
                 double *grids, int ngrids, int *ctr_offsets, int nctr, int *bas,
                 GTOValEnvVars *gto_envs)
{
    dim3 threads(NG_PER_BLOCK);
    dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);

    for (int ictr = 0; ictr < nctr; ictr++){
        int ish = ctr_offsets[ictr];
        const int l =  bas[ANG_OF+ish*BAS_SLOTS];
        int nprim = bas[NPRIM_OF+ish*BAS_SLOTS];
        int bas_offset = ctr_offsets[ictr];
        blocks.y = ctr_offsets[ictr+1] - bas_offset;
        if (blocks.y == 0){
            continue;
        }
        if (l > 8){
            fprintf(stderr, "l = %d not supported\n", l);
            return 1;
        }
        _screen_index<<<blocks, threads, 0, stream>>> (non0shl_idx, cutoff, l, nprim, 
                grids, ngrids, bas_offset, *gto_envs);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of GDFTscreen_index: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

}
