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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"

#define THREADS         256
#define DM_BLOCK        8
#define Ex_at(i,j,t)    Ex[(i)*5+(j)+(t)*25]
#define Ey_at(i,j,t)    Ey[(i)*5+(j)+(t)*25]
#define Ez_at(i,j,t)    Ez[(i)*5+(j)+(t)*25]

__global__ static
void dm_to_Rt_kernel(double *out, double *dm, int n_dm, RysIntEnvVars envs,
                     uint32_t *bas_ij_idx, int *pair_loc, int npairs,
                     int *ao_loc)
{
    int pair_ij = blockIdx.x * blockDim.x + threadIdx.x; 
    if (pair_ij >= npairs) {
        return;
    }
    double *env = envs.env;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
    double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
    double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    double aij = ai + aj;
    double aj_aij = aj / aij;
    double xi = env[ri+0];
    double yi = env[ri+1];
    double zi = env[ri+2];
    double xj = env[rj+0];
    double yj = env[rj+1];
    double zj = env[rj+2];
    double xjxi = xj - xi;
    double yjyi = yj - yi;
    double zjzi = zj - zi;
    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
    double xpa = xjxi * aj_aij;
    double ypa = yjyi * aj_aij;
    double zpa = zjzi * aj_aij;
    double xpb = xpa - xjxi;
    double ypb = ypa - yjyi;
    double zpb = zpa - zjzi;
    double theta_ij = ai * aj / aij;
    double Kab = exp(-theta_ij * rr_ij);
    double cc = Kab * ci * cj;
    if (ish == jsh) {
        cc *= .5;
    }
    double *Rt = out + pair_loc[pair_ij];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    size_t Nao = ao_loc[nbas];
    dm += i0 * Nao + j0;
    size_t dm_xyz_size = pair_loc[npairs];
    size_t Nao2 = Nao * Nao;

    int lij = li + lj;
    double Ex[5*5*9];
    double Ey[5*5*9];
    double Ez[5*5*9];
    Ex_at(0,0,0) = 1.;
    Ey_at(0,0,0) = 1.;
    Ez_at(0,0,0) = cc;
    for (int t = 1; t <= lij; t++) {
        Ex_at(0,0,t) = 0.;
        Ey_at(0,0,t) = 0.;
        Ez_at(0,0,t) = 0.;
    }

    for (int j = 1; j <= lj; j++) {
        Ex_at(0,j,0) = xpb * Ex_at(0,j-1,0) + Ex_at(0,j-1,1);
        Ey_at(0,j,0) = ypb * Ey_at(0,j-1,0) + Ey_at(0,j-1,1);
        Ez_at(0,j,0) = zpb * Ez_at(0,j-1,0) + Ez_at(0,j-1,1);
        for (int t = 1; t <= lij; t++) {
            double fac = j/(2*aij*t);
            Ex_at(0,j,t) = fac * Ex_at(0,j-1,t-1);
            Ey_at(0,j,t) = fac * Ey_at(0,j-1,t-1);
            Ez_at(0,j,t) = fac * Ez_at(0,j-1,t-1);
        }
    }

    for (int i = 1; i <= li; i++) {
        Ex_at(i,0,0) = xpa * Ex_at(i-1,0,0) + Ex_at(i-1,0,1);
        Ey_at(i,0,0) = ypa * Ey_at(i-1,0,0) + Ey_at(i-1,0,1);
        Ez_at(i,0,0) = zpa * Ez_at(i-1,0,0) + Ez_at(i-1,0,1);
        for (int t = 1; t <= lij; t++) {
            double fac = i/(2*aij*t);
            Ex_at(i,0,t) = fac * Ex_at(i-1,0,t-1);
            Ey_at(i,0,t) = fac * Ey_at(i-1,0,t-1);
            Ez_at(i,0,t) = fac * Ez_at(i-1,0,t-1);
        }
    }

    for (int i = 1; i <= li; i++) {
        for (int j = 1; j <= lj; j++) {
            Ex_at(i,j,0) = xpb * Ex_at(i,j-1,0) + Ex_at(i,j-1,1);
            Ey_at(i,j,0) = ypb * Ey_at(i,j-1,0) + Ey_at(i,j-1,1);
            Ez_at(i,j,0) = zpb * Ez_at(i,j-1,0) + Ez_at(i,j-1,1);
            for (int t = 1; t <= lij; t++) {
                double fac = i/(2*aij*t);
                double fac1 = j/(2*aij*t);
                Ex_at(i,j,t) = fac*Ex_at(i-1,j,t-1) + fac1*Ex_at(i,j-1,t-1);
                Ey_at(i,j,t) = fac*Ey_at(i-1,j,t-1) + fac1*Ey_at(i,j-1,t-1);
                Ez_at(i,j,t) = fac*Ez_at(i-1,j,t-1) + fac1*Ez_at(i,j-1,t-1);
            }
        }
    }

    for (int m0 = 0; m0 < n_dm; m0 += DM_BLOCK) {
        int n = 0;
        // products subject to t+u+v <= li+lj
        for (int t = 0; t <= lij; t++) {
        for (int u = 0; u <= lij-t; u++) {
        for (int v = 0; v <= lij-t-u; v++, n++) {
            double res[DM_BLOCK];
#pragma unroll
            for (int m = 0; m < DM_BLOCK; m++) {
                if (m + m0 >= n_dm) break;
                res[m] = 0;
            }
            for (int ix = li, i = 0; ix >= 0; ix--) {
            for (int iy = li-ix; iy >= 0; iy--, i++) {
                int iz = li - ix - iy;
                for (int jx = lj, j = 0; jx >= 0; jx--) {
                for (int jy = lj-jx; jy >= 0; jy--, j++) {
                    int jz = lj - jx - jy;
                    double Et = Ex_at(ix,jx,t) * Ey_at(iy,jy,u) * Ez_at(iz,jz,v);
#pragma unroll
                    for (int m = 0; m < DM_BLOCK; m++) {
                        if (m0 + m >= n_dm) break;
                        res[m] += Et * dm[(m0+m)*Nao2+i*Nao+j];
                    }
                } }
            } }
#pragma unroll
            for (int m = 0; m < DM_BLOCK; m++) {
                if (m0 + m >= n_dm) break;
                Rt[(m0+m)*dm_xyz_size+n] = res[m];
            }
        } } }
    }
}

__global__ static
void Rt_to_dm_kernel(double *dm, double *Rt, int n_dm, RysIntEnvVars envs,
                     uint32_t *bas_ij_idx, int *pair_loc, int npairs,
                     int *ao_loc)
{
    int pair_ij = blockIdx.x * blockDim.x + threadIdx.x; 
    if (pair_ij >= npairs) {
        return;
    }
    double *env = envs.env;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    uint32_t bas_ij = bas_ij_idx[pair_ij];
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
    double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
    double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
    double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
    double aij = ai + aj;
    double aj_aij = aj / aij;
    double xi = env[ri+0];
    double yi = env[ri+1];
    double zi = env[ri+2];
    double xj = env[rj+0];
    double yj = env[rj+1];
    double zj = env[rj+2];
    double xjxi = xj - xi;
    double yjyi = yj - yi;
    double zjzi = zj - zi;
    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
    double xpa = xjxi * aj_aij;
    double ypa = yjyi * aj_aij;
    double zpa = zjzi * aj_aij;
    double xpb = xpa - xjxi;
    double ypb = ypa - yjyi;
    double zpb = zpa - zjzi;
    double theta_ij = ai * aj / aij;
    double Kab = exp(-theta_ij * rr_ij);
    double cc = Kab * ci * cj;
    if (ish == jsh) {
        cc *= .5;
    }
    Rt += pair_loc[pair_ij];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    size_t Nao = ao_loc[nbas];
    dm += i0 * Nao + j0;
    size_t dm_xyz_size = pair_loc[npairs];
    size_t Nao2 = Nao * Nao;

    int lij = li + lj;
    double Ex[5*5*9];
    double Ey[5*5*9];
    double Ez[5*5*9];
    Ex_at(0,0,0) = 1.;
    Ey_at(0,0,0) = 1.;
    Ez_at(0,0,0) = cc;
    for (int t = 1; t <= lij; t++) {
        Ex_at(0,0,t) = 0.;
        Ey_at(0,0,t) = 0.;
        Ez_at(0,0,t) = 0.;
    }

    for (int j = 1; j <= lj; j++) {
        Ex_at(0,j,0) = xpb * Ex_at(0,j-1,0) + Ex_at(0,j-1,1);
        Ey_at(0,j,0) = ypb * Ey_at(0,j-1,0) + Ey_at(0,j-1,1);
        Ez_at(0,j,0) = zpb * Ez_at(0,j-1,0) + Ez_at(0,j-1,1);
        for (int t = 1; t <= lij; t++) {
            double fac = j/(2*aij*t);
            Ex_at(0,j,t) = fac * Ex_at(0,j-1,t-1);
            Ey_at(0,j,t) = fac * Ey_at(0,j-1,t-1);
            Ez_at(0,j,t) = fac * Ez_at(0,j-1,t-1);
        }
    }

    for (int i = 1; i <= li; i++) {
        Ex_at(i,0,0) = xpa * Ex_at(i-1,0,0) + Ex_at(i-1,0,1);
        Ey_at(i,0,0) = ypa * Ey_at(i-1,0,0) + Ey_at(i-1,0,1);
        Ez_at(i,0,0) = zpa * Ez_at(i-1,0,0) + Ez_at(i-1,0,1);
        for (int t = 1; t <= lij; t++) {
            double fac = i/(2*aij*t);
            Ex_at(i,0,t) = fac * Ex_at(i-1,0,t-1);
            Ey_at(i,0,t) = fac * Ey_at(i-1,0,t-1);
            Ez_at(i,0,t) = fac * Ez_at(i-1,0,t-1);
        }
    }

    for (int i = 1; i <= li; i++) {
        for (int j = 1; j <= lj; j++) {
            Ex_at(i,j,0) = xpb * Ex_at(i,j-1,0) + Ex_at(i,j-1,1);
            Ey_at(i,j,0) = ypb * Ey_at(i,j-1,0) + Ey_at(i,j-1,1);
            Ez_at(i,j,0) = zpb * Ez_at(i,j-1,0) + Ez_at(i,j-1,1);
            for (int t = 1; t <= lij; t++) {
                double fac = i/(2*aij*t);
                double fac1 = j/(2*aij*t);
                Ex_at(i,j,t) = fac*Ex_at(i-1,j,t-1) + fac1*Ex_at(i,j-1,t-1);
                Ey_at(i,j,t) = fac*Ey_at(i-1,j,t-1) + fac1*Ey_at(i,j-1,t-1);
                Ez_at(i,j,t) = fac*Ez_at(i-1,j,t-1) + fac1*Ez_at(i,j-1,t-1);
            }
        }
    }

    for (int m0 = 0; m0 < n_dm; m0 += DM_BLOCK) {
        for (int ix = li, i = 0; ix >= 0; ix--) {
        for (int iy = li-ix; iy >= 0; iy--, i++) {
            int iz = li - ix - iy;
            for (int jx = lj, j = 0; jx >= 0; jx--) {
            for (int jy = lj-jx; jy >= 0; jy--, j++) {
                int jz = lj - jx - jy;
                double res[DM_BLOCK];
#pragma unroll
                for (int m = 0; m < DM_BLOCK; m++) {
                    if (m + m0 >= n_dm) break;
                    res[m] = 0;
                }
                int n = 0;
                // products subject to t+u+v <= li+lj
                for (int t = 0; t <= lij; t++) {
                for (int u = 0; u <= lij-t; u++) {
                for (int v = 0; v <= lij-t-u; v++, n++) {
                    double Et = Ex_at(ix,jx,t) * Ey_at(iy,jy,u) * Ez_at(iz,jz,v);
#pragma unroll
                    for (int m = 0; m < DM_BLOCK; m++) {
                        if (m0 + m >= n_dm) break;
                        res[m] += Et * Rt[(m0+m)*dm_xyz_size+n];
                    }
                } } }
#pragma unroll
                for (int m = 0; m < DM_BLOCK; m++) {
                    if (m + m0 >= n_dm) break;
                    dm[(m0+m)*Nao2+i*Nao+j] = res[m];
                }
            } }
        } }
    }
}

__global__ static
void aux_to_Rt_kernel(double *out, double *aux, RysIntEnvVars envs,
                      int *aux_loc, int *aux_xyz_loc, int nbas_aux)
{
    int ksh = blockIdx.x * blockDim.x + threadIdx.x; 
    if (ksh >= nbas_aux) {
        return;
    }
    aux += aux_loc[ksh];
    double *Rt = out + aux_xyz_loc[ksh];
    double *env = envs.env;
    int *bas = envs.bas;
    int nbas = envs.nbas;
    ksh += nbas;
    int lk = bas[ksh*BAS_SLOTS+ANG_OF];
    double ak = env[bas[ksh*BAS_SLOTS+PTR_EXP]];
    double ck = env[bas[ksh*BAS_SLOTS+PTR_COEFF]];

    double aa[6];
    aa[0] = .5 / ak;
    for (int n = 1; n < 6; n++) aa[n] = aa[n-1] * aa[0];

    switch (lk) {
    case 0:
    Rt[0] = ck * (aux[0]);
    break;
    case 1:
    Rt[1] = ck * (aux[2] * aa[0]);
    Rt[2] = ck * (aux[1] * aa[0]);
    Rt[3] = ck * (aux[0] * aa[0]);
    break;
    case 2:
    Rt[0] = ck * (aux[0] * aa[0] + aux[3] * aa[0] + aux[5] * aa[0]);
    Rt[2] = ck * (aux[5] * aa[1]);
    Rt[4] = ck * (aux[4] * aa[1]);
    Rt[5] = ck * (aux[3] * aa[1]);
    Rt[7] = ck * (aux[2] * aa[1]);
    Rt[8] = ck * (aux[1] * aa[1]);
    Rt[9] = ck * (aux[0] * aa[1]);
    break;
    case 3:
    Rt[1] = ck * (aux[2] * aa[1] + aux[7] * aa[1] + aux[9] * aa[1] * 3.0);
    Rt[3] = ck * (aux[9] * aa[2]);
    Rt[4] = ck * (aux[1] * aa[1] + aux[6] * aa[1] * 3.0 + aux[8] * aa[1]);
    Rt[6] = ck * (aux[8] * aa[2]);
    Rt[8] = ck * (aux[7] * aa[2]);
    Rt[9] = ck * (aux[6] * aa[2]);
    Rt[10] = ck * (aux[0] * aa[1] * 3.0 + aux[3] * aa[1] + aux[5] * aa[1]);
    Rt[12] = ck * (aux[5] * aa[2]);
    Rt[14] = ck * (aux[4] * aa[2]);
    Rt[15] = ck * (aux[3] * aa[2]);
    Rt[17] = ck * (aux[2] * aa[2]);
    Rt[18] = ck * (aux[1] * aa[2]);
    Rt[19] = ck * (aux[0] * aa[2]);
    break;
    case 4:
    Rt[0] = ck * (aux[0] * aa[1] * 3.0 + aux[3] * aa[1] + aux[5] * aa[1] + aux[10] * aa[1] * 3.0 + aux[12] * aa[1] + aux[14] * aa[1] * 3.0);
    Rt[2] = ck * (aux[5] * aa[2] + aux[12] * aa[2] + aux[14] * aa[2] * 6.0);
    Rt[4] = ck * (aux[14] * aa[3]);
    Rt[6] = ck * (aux[4] * aa[2] + aux[11] * aa[2] * 3.0 + aux[13] * aa[2] * 3.0);
    Rt[8] = ck * (aux[13] * aa[3]);
    Rt[9] = ck * (aux[3] * aa[2] + aux[10] * aa[2] * 6.0 + aux[12] * aa[2]);
    Rt[11] = ck * (aux[12] * aa[3]);
    Rt[13] = ck * (aux[11] * aa[3]);
    Rt[14] = ck * (aux[10] * aa[3]);
    Rt[16] = ck * (aux[2] * aa[2] * 3.0 + aux[7] * aa[2] + aux[9] * aa[2] * 3.0);
    Rt[18] = ck * (aux[9] * aa[3]);
    Rt[19] = ck * (aux[1] * aa[2] * 3.0 + aux[6] * aa[2] * 3.0 + aux[8] * aa[2]);
    Rt[21] = ck * (aux[8] * aa[3]);
    Rt[23] = ck * (aux[7] * aa[3]);
    Rt[24] = ck * (aux[6] * aa[3]);
    Rt[25] = ck * (aux[0] * aa[2] * 6.0 + aux[3] * aa[2] + aux[5] * aa[2]);
    Rt[27] = ck * (aux[5] * aa[3]);
    Rt[29] = ck * (aux[4] * aa[3]);
    Rt[30] = ck * (aux[3] * aa[3]);
    Rt[32] = ck * (aux[2] * aa[3]);
    Rt[33] = ck * (aux[1] * aa[3]);
    Rt[34] = ck * (aux[0] * aa[3]);
    break;
    case 5:
    Rt[1] = ck * (aux[2] * aa[2] * 3.0 + aux[7] * aa[2] + aux[9] * aa[2] * 3.0 + aux[16] * aa[2] * 3.0 + aux[18] * aa[2] * 3.0 + aux[20] * aa[2] * 15.0);
    Rt[3] = ck * (aux[9] * aa[3] + aux[18] * aa[3] + aux[20] * aa[3] * 10.0);
    Rt[5] = ck * (aux[20] * aa[4]);
    Rt[6] = ck * (aux[1] * aa[2] * 3.0 + aux[6] * aa[2] * 3.0 + aux[8] * aa[2] + aux[15] * aa[2] * 15.0 + aux[17] * aa[2] * 3.0 + aux[19] * aa[2] * 3.0);
    Rt[8] = ck * (aux[8] * aa[3] + aux[17] * aa[3] * 3.0 + aux[19] * aa[3] * 6.0);
    Rt[10] = ck * (aux[19] * aa[4]);
    Rt[12] = ck * (aux[7] * aa[3] + aux[16] * aa[3] * 6.0 + aux[18] * aa[3] * 3.0);
    Rt[14] = ck * (aux[18] * aa[4]);
    Rt[15] = ck * (aux[6] * aa[3] + aux[15] * aa[3] * 10.0 + aux[17] * aa[3]);
    Rt[17] = ck * (aux[17] * aa[4]);
    Rt[19] = ck * (aux[16] * aa[4]);
    Rt[20] = ck * (aux[15] * aa[4]);
    Rt[21] = ck * (aux[0] * aa[2] * 15.0 + aux[3] * aa[2] * 3.0 + aux[5] * aa[2] * 3.0 + aux[10] * aa[2] * 3.0 + aux[12] * aa[2] + aux[14] * aa[2] * 3.0);
    Rt[23] = ck * (aux[5] * aa[3] * 3.0 + aux[12] * aa[3] + aux[14] * aa[3] * 6.0);
    Rt[25] = ck * (aux[14] * aa[4]);
    Rt[27] = ck * (aux[4] * aa[3] * 3.0 + aux[11] * aa[3] * 3.0 + aux[13] * aa[3] * 3.0);
    Rt[29] = ck * (aux[13] * aa[4]);
    Rt[30] = ck * (aux[3] * aa[3] * 3.0 + aux[10] * aa[3] * 6.0 + aux[12] * aa[3]);
    Rt[32] = ck * (aux[12] * aa[4]);
    Rt[34] = ck * (aux[11] * aa[4]);
    Rt[35] = ck * (aux[10] * aa[4]);
    Rt[37] = ck * (aux[2] * aa[3] * 6.0 + aux[7] * aa[3] + aux[9] * aa[3] * 3.0);
    Rt[39] = ck * (aux[9] * aa[4]);
    Rt[40] = ck * (aux[1] * aa[3] * 6.0 + aux[6] * aa[3] * 3.0 + aux[8] * aa[3]);
    Rt[42] = ck * (aux[8] * aa[4]);
    Rt[44] = ck * (aux[7] * aa[4]);
    Rt[45] = ck * (aux[6] * aa[4]);
    Rt[46] = ck * (aux[0] * aa[3] * 10.0 + aux[3] * aa[3] + aux[5] * aa[3]);
    Rt[48] = ck * (aux[5] * aa[4]);
    Rt[50] = ck * (aux[4] * aa[4]);
    Rt[51] = ck * (aux[3] * aa[4]);
    Rt[53] = ck * (aux[2] * aa[4]);
    Rt[54] = ck * (aux[1] * aa[4]);
    Rt[55] = ck * (aux[0] * aa[4]);
    break;
    case 6:
    Rt[0] = ck * (aux[0] * aa[2] * 15.0 + aux[3] * aa[2] * 3.0 + aux[5] * aa[2] * 3.0 + aux[10] * aa[2] * 3.0 + aux[12] * aa[2] + aux[14] * aa[2] * 3.0 + aux[21] * aa[2] * 15.0 + aux[23] * aa[2] * 3.0 + aux[25] * aa[2] * 3.0 + aux[27] * aa[2] * 15.0);
    Rt[2] = ck * (aux[5] * aa[3] * 3.0 + aux[12] * aa[3] + aux[14] * aa[3] * 6.0 + aux[23] * aa[3] * 3.0 + aux[25] * aa[3] * 6.0 + aux[27] * aa[3] * 45.0);
    Rt[4] = ck * (aux[14] * aa[4] + aux[25] * aa[4] + aux[27] * aa[4] * 15.0);
    Rt[6] = ck * (aux[27] * aa[5]);
    Rt[8] = ck * (aux[4] * aa[3] * 3.0 + aux[11] * aa[3] * 3.0 + aux[13] * aa[3] * 3.0 + aux[22] * aa[3] * 15.0 + aux[24] * aa[3] * 9.0 + aux[26] * aa[3] * 15.0);
    Rt[10] = ck * (aux[13] * aa[4] + aux[24] * aa[4] * 3.0 + aux[26] * aa[4] * 10.0);
    Rt[12] = ck * (aux[26] * aa[5]);
    Rt[13] = ck * (aux[3] * aa[3] * 3.0 + aux[10] * aa[3] * 6.0 + aux[12] * aa[3] + aux[21] * aa[3] * 45.0 + aux[23] * aa[3] * 6.0 + aux[25] * aa[3] * 3.0);
    Rt[15] = ck * (aux[12] * aa[4] + aux[23] * aa[4] * 6.0 + aux[25] * aa[4] * 6.0);
    Rt[17] = ck * (aux[25] * aa[5]);
    Rt[19] = ck * (aux[11] * aa[4] + aux[22] * aa[4] * 10.0 + aux[24] * aa[4] * 3.0);
    Rt[21] = ck * (aux[24] * aa[5]);
    Rt[22] = ck * (aux[10] * aa[4] + aux[21] * aa[4] * 15.0 + aux[23] * aa[4]);
    Rt[24] = ck * (aux[23] * aa[5]);
    Rt[26] = ck * (aux[22] * aa[5]);
    Rt[27] = ck * (aux[21] * aa[5]);
    Rt[29] = ck * (aux[2] * aa[3] * 15.0 + aux[7] * aa[3] * 3.0 + aux[9] * aa[3] * 9.0 + aux[16] * aa[3] * 3.0 + aux[18] * aa[3] * 3.0 + aux[20] * aa[3] * 15.0);
    Rt[31] = ck * (aux[9] * aa[4] * 3.0 + aux[18] * aa[4] + aux[20] * aa[4] * 10.0);
    Rt[33] = ck * (aux[20] * aa[5]);
    Rt[34] = ck * (aux[1] * aa[3] * 15.0 + aux[6] * aa[3] * 9.0 + aux[8] * aa[3] * 3.0 + aux[15] * aa[3] * 15.0 + aux[17] * aa[3] * 3.0 + aux[19] * aa[3] * 3.0);
    Rt[36] = ck * (aux[8] * aa[4] * 3.0 + aux[17] * aa[4] * 3.0 + aux[19] * aa[4] * 6.0);
    Rt[38] = ck * (aux[19] * aa[5]);
    Rt[40] = ck * (aux[7] * aa[4] * 3.0 + aux[16] * aa[4] * 6.0 + aux[18] * aa[4] * 3.0);
    Rt[42] = ck * (aux[18] * aa[5]);
    Rt[43] = ck * (aux[6] * aa[4] * 3.0 + aux[15] * aa[4] * 10.0 + aux[17] * aa[4]);
    Rt[45] = ck * (aux[17] * aa[5]);
    Rt[47] = ck * (aux[16] * aa[5]);
    Rt[48] = ck * (aux[15] * aa[5]);
    Rt[49] = ck * (aux[0] * aa[3] * 45.0 + aux[3] * aa[3] * 6.0 + aux[5] * aa[3] * 6.0 + aux[10] * aa[3] * 3.0 + aux[12] * aa[3] + aux[14] * aa[3] * 3.0);
    Rt[51] = ck * (aux[5] * aa[4] * 6.0 + aux[12] * aa[4] + aux[14] * aa[4] * 6.0);
    Rt[53] = ck * (aux[14] * aa[5]);
    Rt[55] = ck * (aux[4] * aa[4] * 6.0 + aux[11] * aa[4] * 3.0 + aux[13] * aa[4] * 3.0);
    Rt[57] = ck * (aux[13] * aa[5]);
    Rt[58] = ck * (aux[3] * aa[4] * 6.0 + aux[10] * aa[4] * 6.0 + aux[12] * aa[4]);
    Rt[60] = ck * (aux[12] * aa[5]);
    Rt[62] = ck * (aux[11] * aa[5]);
    Rt[63] = ck * (aux[10] * aa[5]);
    Rt[65] = ck * (aux[2] * aa[4] * 10.0 + aux[7] * aa[4] + aux[9] * aa[4] * 3.0);
    Rt[67] = ck * (aux[9] * aa[5]);
    Rt[68] = ck * (aux[1] * aa[4] * 10.0 + aux[6] * aa[4] * 3.0 + aux[8] * aa[4]);
    Rt[70] = ck * (aux[8] * aa[5]);
    Rt[72] = ck * (aux[7] * aa[5]);
    Rt[73] = ck * (aux[6] * aa[5]);
    Rt[74] = ck * (aux[0] * aa[4] * 15.0 + aux[3] * aa[4] + aux[5] * aa[4]);
    Rt[76] = ck * (aux[5] * aa[5]);
    Rt[78] = ck * (aux[4] * aa[5]);
    Rt[79] = ck * (aux[3] * aa[5]);
    Rt[81] = ck * (aux[2] * aa[5]);
    Rt[82] = ck * (aux[1] * aa[5]);
    Rt[83] = ck * (aux[0] * aa[5]);
    break;
    }
}

extern "C" {
int dm_to_Rt(double *out, double *dm, int n_dm, RysIntEnvVars *envs,
             uint32_t *bas_ij_idx, int *pair_loc, int npairs, int *ao_loc)
{
    int blocks = (npairs + THREADS - 1) / THREADS;
    dm_to_Rt_kernel<<<blocks, THREADS>>>(out, dm, n_dm, *envs, bas_ij_idx, pair_loc, npairs, ao_loc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in dm_to_Rt_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int Rt_to_dm(double *dm, double *Rt, int n_dm, RysIntEnvVars *envs,
             uint32_t *bas_ij_idx, int *pair_loc, int npairs, int *ao_loc)
{
    int blocks = (npairs + THREADS - 1) / THREADS;
    Rt_to_dm_kernel<<<blocks, THREADS>>>(dm, Rt, n_dm, *envs, bas_ij_idx, pair_loc, npairs, ao_loc);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in Rt_to_dm_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int aux_to_Rt(double *out, double *aux, RysIntEnvVars *envs,
              int *aux_loc, int *aux_xyz_loc, int nbas_aux)
{
    int blocks = (nbas_aux + THREADS - 1) / THREADS;
    aux_to_Rt_kernel<<<blocks, THREADS>>>(out, aux, *envs, aux_loc, aux_xyz_loc, nbas_aux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in aux_to_Rt_kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
