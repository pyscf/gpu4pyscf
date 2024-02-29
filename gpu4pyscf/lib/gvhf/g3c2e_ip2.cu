/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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


template <int NROOTS, int GSIZE> __global__
void GINTint3c2e_ip2_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
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
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double* __restrict__ exp = c_bpcache.a1;
    double uw[NROOTS*2];
    double g[2*GSIZE];
    double *f = g + GSIZE;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;

    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }

    double j3[GPU_CART_MAX * 3];
    double k3[GPU_CART_MAX * 3];
    for (int k = 0; k < GPU_CART_MAX * 3; k++){
        j3[k] = 0.0;
        k3[k] = 0.0;
    }
    if (active) {
        for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {

            double aij = a12[ij];
            double xij = x12[ij];
            double yij = y12[ij];
            double zij = z12[ij];
            double akl = a12[kl];
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
            GINTrys_root<NROOTS>(x, uw);
            GINTscale_u<NROOTS>(uw, theta);
            GINTg0_2e_2d4d<NROOTS>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

            double ak2 = -2.0*exp[kl];
            GINTnabla1k_2e<NROOTS>(envs, f, g, ak2, envs.i_l, envs.j_l, envs.k_l);
            GINTkernel_int3c2e_ip2_getjk_direct<NROOTS>(envs, jk, j3, k3, f, g, ish, jsh, ksh);
            }
        }
    }

    write_int3c2e_ip2_jk(jk, j3, k3, ksh);
}


__global__
static void GINTrun_int3c2e_ip2_jk_kernel0010(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
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
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
        double ak2 = -2.0*a1[kl];
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
        double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
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
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c0px;
        double g_2 = 1;
        double g_3 = c0py;
        double g_4 = weight0 * fac;
        double g_5 = c0pz * g_4;

        double f_1 = ak2 * g_1;
        double f_3 = ak2 * g_3;
        double f_5 = ak2 * g_5;

        gout0 += f_1 * g_2 * g_4;
        gout1 += g_0 * f_3 * g_4;
        gout2 += g_0 * g_2 * f_5;

    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish] - jk.ao_offsets_i;
    int j0 = ao_loc[jsh] - jk.ao_offsets_j;
    int k0 = ao_loc[ksh] - jk.ao_offsets_k;

    int nao = jk.nao;
    int naux = jk.naux;
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
        vj_tmp[0] = gout0 * rhoj_tmp;
        vj_tmp[1] = gout1 * rhoj_tmp;
        vj_tmp[2] = gout2 * rhoj_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vj_tmp[j]; __syncthreads();
            if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
            if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
            if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
            if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
            if (tx == 0) atomicAdd(vj+k0+j*naux, sdata[0][ty]);
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
            if(tx<8) sdata[tx][ty] += sdata[tx+8][ty]; __syncthreads();
            if(tx<4) sdata[tx][ty] += sdata[tx+4][ty]; __syncthreads();
            if(tx<2) sdata[tx][ty] += sdata[tx+2][ty]; __syncthreads();
            if(tx<1) sdata[tx][ty] += sdata[tx+1][ty]; __syncthreads();
            if (tx == 0) atomicAdd(vk+k0+j*naux, sdata[0][ty]);
        }
    }
}