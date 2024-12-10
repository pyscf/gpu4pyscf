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

// Unrolled version
template <int LI, int LJ, int LK> __global__
void GINTint3c2e_ip1_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
    }
    double norm = envs.fac;
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    
    double* __restrict__ exp = c_bpcache.a1;
    constexpr int LI_CEIL = LI + 1;
    constexpr int NROOTS = (LI_CEIL+LJ+LK)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI_CEIL+1)*(LJ+1)*(LK+1);

    double g[2*GSIZE];
    double *f = g + GSIZE;

    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    constexpr int nfi = (LI+1)*(LI+2)/2;
    double j3[nfi * 3];
    double k3[nfi * 3];
    for (int k = 0; k < nfi * 3; k++){
        j3[k] = 0.0;
        k3[k] = 0.0;
    }
    if (active) {
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
                GINTg0_int3c2e<LI_CEIL, LJ, LK>(envs, g, norm, as_ish, as_jsh, ksh, ij, kl);
                double ai2 = -2.0*exp[ij];
                GINTnabla1i_2e<LI, LJ, LK, NROOTS>(envs, f, g, ai2);
                GINTkernel_int3c2e_ip1_getjk_direct<LI, LJ, LK>(envs, jk, j3, k3, f, g, ish, jsh, ksh);
            }
        }
    }

    write_int3c2e_ip1_jk(jk, j3, k3, ish);
}

template <int NROOTS, int GSIZE> __global__
void GINTint3c2e_ip1_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
    }
    double norm = envs.fac;
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    
    double* __restrict__ exp = c_bpcache.a1;
    double g[2*GSIZE];
    double * __restrict__ f = g + GSIZE;

    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    double j3[GPU_AO_NF * 3];
    double k3[GPU_AO_NF * 3];
    for (int k = 0; k < GPU_AO_NF * 3; k++){
        j3[k] = 0.0;
        k3[k] = 0.0;
    }
    if (active) {
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_int3c2e<NROOTS>(envs, g, norm, as_ish, as_jsh, ksh, ij, kl);
            double ai2 = -2.0*exp[ij];
            GINTnabla1i_2e<NROOTS>(envs, f, g, ai2, envs.i_l, envs.j_l, envs.k_l);
            GINTkernel_int3c2e_ip1_getjk_direct<NROOTS>(envs, jk, j3, k3, f, g, ish, jsh, ksh);
            }
        }
    }

    write_int3c2e_ip1_jk(jk, j3, k3, ish);
}

__global__
static void GINTint3c2e_ip1_jk_kernel000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double* __restrict__ a1 = c_bpcache.a1;
    int ij, kl;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
        double ai2 = -2.0*a1[ij];
        double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        double akl = a12[kl];
        double ekl = e12[kl];
        double xkl = x12[kl];
        double ykl = y12[kl];
        double zkl = z12[kl];
        double xijxkl = xij - xkl;
        double yijykl = yij - ykl;
        double zijzkl = zij - zkl;
        double aijkl = aij + akl;
        double a1 = aij * akl;
        double a0 = a1 / aijkl;
        double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
        a0 *= theta;
        double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
        double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
        double root0, weight0;
        if (x < 3.e-7) {
            root0 = 0.5;
            weight0 = 1.;
        } else {
            double tt = sqrt(x);
            double fmt0 = SQRTPIE4 / tt * erf(tt);
            weight0 = fmt0;
            double e = exp(-x);
            double b = .5 / x;
            double fmt1 = b * (fmt0 - e);
            root0 = fmt1 / (fmt0 - fmt1);
        }
        root0 /= root0 + 1 - root0 * theta;
        double u2 = a0 * root0;
        double tmp2 = akl * u2 / (u2 * aijkl + a1);;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = 1;
        double g_3 = c00y;
        double g_4 = norm * fac * weight0;
        double g_5 = g_4 * c00z;

        double f_1 = ai2 * g_1;
        double f_3 = ai2 * g_3;
        double f_5 = ai2 * g_5;

        gout0 += f_1 * g_2 * g_4;
        gout1 += g_0 * f_3 * g_4;
        gout2 += g_0 * g_2 * f_5;
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh] - jk.ao_offsets_k;

    int nao = jk.nao;
    double* __restrict__ dm = jk.dm;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ vj = jk.vj;
    double* __restrict__ vk = jk.vk;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];
    if (!active){
        gout0 = 0.0; gout1 = 0.0; gout2 = 0.0;
    }
    if (vj != NULL){
        double rhoj_tmp;
        int off_dm = i0 + nao*j0;
        rhoj_tmp = dm[off_dm] * rhoj[k0];
        double vj_tmp[3];
        vj_tmp[0] = gout0*rhoj_tmp;
        vj_tmp[1] = gout1*rhoj_tmp;
        vj_tmp[2] = gout2*rhoj_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vj_tmp[j]; __syncthreads();
            if(THREADSY >= 16 && ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
            if(THREADSY >= 8  && ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
            if(THREADSY >= 4  && ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
            if(THREADSY >= 2  && ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
            if (ty == 0) atomicAdd(vj+i0+j*nao, sdata[tx][0]);
        }
    }
    if (vk != NULL){
        double rhok_tmp;
        int off_rhok = i0 + nao*j0 + k0*nao*nao;
        rhok_tmp = rhok[off_rhok];
        double vk_tmp[3];
        vk_tmp[0] = gout0 * rhok_tmp;
        vk_tmp[1] = gout1 * rhok_tmp;
        vk_tmp[2] = gout2 * rhok_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vk_tmp[j]; __syncthreads();
            if(THREADSY >= 16 && ty<8) sdata[tx][ty] += sdata[tx][ty+8]; __syncthreads();
            if(THREADSY >=  8 && ty<4) sdata[tx][ty] += sdata[tx][ty+4]; __syncthreads();
            if(THREADSY >=  4 && ty<2) sdata[tx][ty] += sdata[tx][ty+2]; __syncthreads();
            if(THREADSY >=  2 && ty<1) sdata[tx][ty] += sdata[tx][ty+1]; __syncthreads();
            if (ty == 0) atomicAdd(vk+i0+j*nao, sdata[tx][0]);
        }
    }
}
