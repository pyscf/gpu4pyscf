/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
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


__global__
static void GINTint2e_jk_kernel1010(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        task_ij = 0; task_kl = 0;
        active = false;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        active = false;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    if(active){
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double tmp3 = tmp1 * aij;
            double c0px = xkl - xk + tmp3 * xijxkl;
            double c0py = ykl - yk + tmp3 * yijykl;
            double c0pz = zkl - zk + tmp3 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c0px;
            double g_3 = c0px * c00x + b00;
            double g_4 = 1;
            double g_5 = c00y;
            double g_6 = c0py;
            double g_7 = c0py * c00y + b00;
            double g_8 = weight0 * fac;
            double g_9 = c00z * g_8;
            double g_10 = c0pz * g_8;
            double g_11 = b00 * g_8 + c0pz * g_9;
            gout0 += g_3 * g_4 * g_8;
            gout1 += g_2 * g_5 * g_8;
            gout2 += g_2 * g_4 * g_9;
            gout3 += g_1 * g_6 * g_8;
            gout4 += g_0 * g_7 * g_8;
            gout5 += g_0 * g_6 * g_9;
            gout6 += g_1 * g_4 * g_10;
            gout7 += g_0 * g_5 * g_10;
            gout8 += g_0 * g_4 * g_11;
        }
    } }
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            //atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            //atomicAdd(vj+(k0+1)+nao*(l0+0), gout3*d_0 + gout4*d_1 + gout5*d_2);
            //atomicAdd(vj+(k0+2)+nao*(l0+0), gout6*d_0 + gout7*d_1 + gout8*d_2);
            block_reduce_x<THREADSX, THREADSY>(gout0*d_0 + gout1*d_1 + gout2*d_2, vj+(k0+0)+nao*(l0+0), tx, ty);
            block_reduce_x<THREADSX, THREADSY>(gout3*d_0 + gout4*d_1 + gout5*d_2, vj+(k0+1)+nao*(l0+0), tx, ty);
            block_reduce_x<THREADSX, THREADSY>(gout6*d_0 + gout7*d_1 + gout8*d_2, vj+(k0+2)+nao*(l0+0), tx, ty);

            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+1)+nao*(l0+0)];
            d_2 = dm[(k0+2)+nao*(l0+0)];
            //atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0 + gout3*d_1 + gout6*d_2);
            //atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0 + gout4*d_1 + gout7*d_2);
            //atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0 + gout5*d_1 + gout8*d_2);
            block_reduce_y<THREADSX, THREADSY>(gout0*d_0 + gout3*d_1 + gout6*d_2, vj+(i0+0)+nao*(j0+0), tx, ty);
            block_reduce_y<THREADSX, THREADSY>(gout1*d_0 + gout4*d_1 + gout7*d_2, vj+(i0+1)+nao*(j0+0), tx, ty);
            block_reduce_y<THREADSX, THREADSY>(gout2*d_0 + gout5*d_1 + gout8*d_2, vj+(i0+2)+nao*(j0+0), tx, ty);

            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout3*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+1), gout4*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+1), gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout6*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+2), gout7*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+2), gout8*d_0);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+0)+nao*(k0+1)];
            d_2 = dm[(j0+0)+nao*(k0+2)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0 + gout3*d_1 + gout6*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0 + gout4*d_1 + gout7*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0 + gout5*d_1 + gout8*d_2);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout3*d_0 + gout4*d_1 + gout5*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout6*d_0 + gout7*d_1 + gout8*d_2);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+0)+nao*(k0+1)];
            d_4 = dm[(i0+1)+nao*(k0+1)];
            d_5 = dm[(i0+2)+nao*(k0+1)];
            d_6 = dm[(i0+0)+nao*(k0+2)];
            d_7 = dm[(i0+1)+nao*(k0+2)];
            d_8 = dm[(i0+2)+nao*(k0+2)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8);
            vk += nao2;
        }
        dm += nao2;
    }
}


__global__
static void GINTint2e_jk_kernel1011(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);
        
        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double tmp3 = tmp1 * aij;
            double b01 = b00 + tmp4 * aij;
            double c0px = xkl - xk + tmp3 * xijxkl;
            double c0py = ykl - yk + tmp3 * yijykl;
            double c0pz = zkl - zk + tmp3 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c0px;
            double g_3 = c0px * c00x + b00;
            double g_4 = c0px + xkxl;
            double g_5 = c00x * (c0px + xkxl) + b00;
            double g_6 = c0px * (c0px + xkxl) + b01;
            double g_7 = b00 * c0px + b01 * c00x + c0px * g_3 + xkxl * g_3;
            double g_8 = 1;
            double g_9 = c00y;
            double g_10 = c0py;
            double g_11 = c0py * c00y + b00;
            double g_12 = c0py + ykyl;
            double g_13 = c00y * (c0py + ykyl) + b00;
            double g_14 = c0py * (c0py + ykyl) + b01;
            double g_15 = b00 * c0py + b01 * c00y + c0py * g_11 + ykyl * g_11;
            double g_16 = weight0 * fac;
            double g_17 = c00z * g_16;
            double g_18 = c0pz * g_16;
            double g_19 = b00 * g_16 + c0pz * g_17;
            double g_20 = g_16 * (c0pz + zkzl);
            double g_21 = b00 * g_16 + c0pz * g_17 + zkzl * g_17;
            double g_22 = b01 * g_16 + c0pz * g_18 + zkzl * g_18;
            double g_23 = b00 * g_18 + b01 * g_17 + c0pz * g_19 + zkzl * g_19;
            gout0 += g_7 * g_8 * g_16;
            gout1 += g_6 * g_9 * g_16;
            gout2 += g_6 * g_8 * g_17;
            gout3 += g_5 * g_10 * g_16;
            gout4 += g_4 * g_11 * g_16;
            gout5 += g_4 * g_10 * g_17;
            gout6 += g_5 * g_8 * g_18;
            gout7 += g_4 * g_9 * g_18;
            gout8 += g_4 * g_8 * g_19;
            gout9 += g_3 * g_12 * g_16;
            gout10 += g_2 * g_13 * g_16;
            gout11 += g_2 * g_12 * g_17;
            gout12 += g_1 * g_14 * g_16;
            gout13 += g_0 * g_15 * g_16;
            gout14 += g_0 * g_14 * g_17;
            gout15 += g_1 * g_12 * g_18;
            gout16 += g_0 * g_13 * g_18;
            gout17 += g_0 * g_12 * g_19;
            gout18 += g_3 * g_8 * g_20;
            gout19 += g_2 * g_9 * g_20;
            gout20 += g_2 * g_8 * g_21;
            gout21 += g_1 * g_10 * g_20;
            gout22 += g_0 * g_11 * g_20;
            gout23 += g_0 * g_10 * g_21;
            gout24 += g_1 * g_8 * g_22;
            gout25 += g_0 * g_9 * g_22;
            gout26 += g_0 * g_8 * g_23;
        }
    } }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            atomicAdd(vj+(k0+1)+nao*(l0+0), gout3*d_0 + gout4*d_1 + gout5*d_2);
            atomicAdd(vj+(k0+2)+nao*(l0+0), gout6*d_0 + gout7*d_1 + gout8*d_2);
            atomicAdd(vj+(k0+0)+nao*(l0+1), gout9*d_0 + gout10*d_1 + gout11*d_2);
            atomicAdd(vj+(k0+1)+nao*(l0+1), gout12*d_0 + gout13*d_1 + gout14*d_2);
            atomicAdd(vj+(k0+2)+nao*(l0+1), gout15*d_0 + gout16*d_1 + gout17*d_2);
            atomicAdd(vj+(k0+0)+nao*(l0+2), gout18*d_0 + gout19*d_1 + gout20*d_2);
            atomicAdd(vj+(k0+1)+nao*(l0+2), gout21*d_0 + gout22*d_1 + gout23*d_2);
            atomicAdd(vj+(k0+2)+nao*(l0+2), gout24*d_0 + gout25*d_1 + gout26*d_2);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+1)+nao*(l0+0)];
            d_2 = dm[(k0+2)+nao*(l0+0)];
            d_3 = dm[(k0+0)+nao*(l0+1)];
            d_4 = dm[(k0+1)+nao*(l0+1)];
            d_5 = dm[(k0+2)+nao*(l0+1)];
            d_6 = dm[(k0+0)+nao*(l0+2)];
            d_7 = dm[(k0+1)+nao*(l0+2)];
            d_8 = dm[(k0+2)+nao*(l0+2)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0 + gout3*d_1 + gout6*d_2 + gout9*d_3 + gout12*d_4 + gout15*d_5 + gout18*d_6 + gout21*d_7 + gout24*d_8);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0 + gout4*d_1 + gout7*d_2 + gout10*d_3 + gout13*d_4 + gout16*d_5 + gout19*d_6 + gout22*d_7 + gout25*d_8);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0 + gout5*d_1 + gout8*d_2 + gout11*d_3 + gout14*d_4 + gout17*d_5 + gout20*d_6 + gout23*d_7 + gout26*d_8);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            d_1 = dm[(j0+0)+nao*(l0+1)];
            d_2 = dm[(j0+0)+nao*(l0+2)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0 + gout9*d_1 + gout18*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0 + gout10*d_1 + gout19*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0 + gout11*d_1 + gout20*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout3*d_0 + gout12*d_1 + gout21*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+1), gout4*d_0 + gout13*d_1 + gout22*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+1), gout5*d_0 + gout14*d_1 + gout23*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout6*d_0 + gout15*d_1 + gout24*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+2), gout7*d_0 + gout16*d_1 + gout25*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+2), gout8*d_0 + gout17*d_1 + gout26*d_2);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+0)+nao*(k0+1)];
            d_2 = dm[(j0+0)+nao*(k0+2)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0 + gout3*d_1 + gout6*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0 + gout4*d_1 + gout7*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0 + gout5*d_1 + gout8*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+1), gout9*d_0 + gout12*d_1 + gout15*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+1), gout10*d_0 + gout13*d_1 + gout16*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+1), gout11*d_0 + gout14*d_1 + gout17*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+2), gout18*d_0 + gout21*d_1 + gout24*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+2), gout19*d_0 + gout22*d_1 + gout25*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+2), gout20*d_0 + gout23*d_1 + gout26*d_2);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            d_3 = dm[(i0+0)+nao*(l0+1)];
            d_4 = dm[(i0+1)+nao*(l0+1)];
            d_5 = dm[(i0+2)+nao*(l0+1)];
            d_6 = dm[(i0+0)+nao*(l0+2)];
            d_7 = dm[(i0+1)+nao*(l0+2)];
            d_8 = dm[(i0+2)+nao*(l0+2)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout9*d_3 + gout10*d_4 + gout11*d_5 + gout18*d_6 + gout19*d_7 + gout20*d_8);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout3*d_0 + gout4*d_1 + gout5*d_2 + gout12*d_3 + gout13*d_4 + gout14*d_5 + gout21*d_6 + gout22*d_7 + gout23*d_8);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout6*d_0 + gout7*d_1 + gout8*d_2 + gout15*d_3 + gout16*d_4 + gout17*d_5 + gout24*d_6 + gout25*d_7 + gout26*d_8);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+0)+nao*(k0+1)];
            d_4 = dm[(i0+1)+nao*(k0+1)];
            d_5 = dm[(i0+2)+nao*(k0+1)];
            d_6 = dm[(i0+0)+nao*(k0+2)];
            d_7 = dm[(i0+1)+nao*(k0+2)];
            d_8 = dm[(i0+2)+nao*(k0+2)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8);
            atomicAdd(vk+(j0+0)+nao*(l0+1), gout9*d_0 + gout10*d_1 + gout11*d_2 + gout12*d_3 + gout13*d_4 + gout14*d_5 + gout15*d_6 + gout16*d_7 + gout17*d_8);
            atomicAdd(vk+(j0+0)+nao*(l0+2), gout18*d_0 + gout19*d_1 + gout20*d_2 + gout21*d_3 + gout22*d_4 + gout23*d_5 + gout24*d_6 + gout25*d_7 + gout26*d_8);
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_jk_kernel1100(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double b10 = b00 + tmp4 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c00x + xixj;
            double g_3 = c00x * (c00x + xixj) + b10;
            double g_4 = 1;
            double g_5 = c00y;
            double g_6 = c00y + yiyj;
            double g_7 = c00y * (c00y + yiyj) + b10;
            double g_8 = weight0 * fac;
            double g_9 = c00z * g_8;
            double g_10 = g_8 * (c00z + zizj);
            double g_11 = b10 * g_8 + c00z * g_9 + zizj * g_9;
            gout0 += g_3 * g_4 * g_8;
            gout1 += g_2 * g_5 * g_8;
            gout2 += g_2 * g_4 * g_9;
            gout3 += g_1 * g_6 * g_8;
            gout4 += g_0 * g_7 * g_8;
            gout5 += g_0 * g_6 * g_9;
            gout6 += g_1 * g_4 * g_10;
            gout7 += g_0 * g_5 * g_10;
            gout8 += g_0 * g_4 * g_11;
        }
    } }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            d_3 = dm[(i0+0)+nao*(j0+1)];
            d_4 = dm[(i0+1)+nao*(j0+1)];
            d_5 = dm[(i0+2)+nao*(j0+1)];
            d_6 = dm[(i0+0)+nao*(j0+2)];
            d_7 = dm[(i0+1)+nao*(j0+2)];
            d_8 = dm[(i0+2)+nao*(j0+2)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+1), gout3*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+1), gout4*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+1), gout5*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+2), gout6*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+2), gout7*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+2), gout8*d_0);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            d_1 = dm[(j0+1)+nao*(l0+0)];
            d_2 = dm[(j0+2)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0 + gout3*d_1 + gout6*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0 + gout4*d_1 + gout7*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0 + gout5*d_1 + gout8*d_2);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+1)+nao*(k0+0)];
            d_2 = dm[(j0+2)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0 + gout3*d_1 + gout6*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0 + gout4*d_1 + gout7*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0 + gout5*d_1 + gout8*d_2);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            atomicAdd(vk+(j0+1)+nao*(k0+0), gout3*d_0 + gout4*d_1 + gout5*d_2);
            atomicAdd(vk+(j0+2)+nao*(k0+0), gout6*d_0 + gout7*d_1 + gout8*d_2);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            atomicAdd(vk+(j0+1)+nao*(l0+0), gout3*d_0 + gout4*d_1 + gout5*d_2);
            atomicAdd(vk+(j0+2)+nao*(l0+0), gout6*d_0 + gout7*d_1 + gout8*d_2);
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_jk_kernel1110(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double b10 = b00 + tmp4 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double tmp3 = tmp1 * aij;
            double c0px = xkl - xk + tmp3 * xijxkl;
            double c0py = ykl - yk + tmp3 * yijykl;
            double c0pz = zkl - zk + tmp3 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c00x + xixj;
            double g_3 = c00x * (c00x + xixj) + b10;
            double g_4 = c0px;
            double g_5 = c0px * c00x + b00;
            double g_6 = c0px * (c00x + xixj) + b00;
            double g_7 = b00 * c00x + b10 * c0px + c00x * g_5 + xixj * g_5;
            double g_8 = 1;
            double g_9 = c00y;
            double g_10 = c00y + yiyj;
            double g_11 = c00y * (c00y + yiyj) + b10;
            double g_12 = c0py;
            double g_13 = c0py * c00y + b00;
            double g_14 = c0py * (c00y + yiyj) + b00;
            double g_15 = b00 * c00y + b10 * c0py + c00y * g_13 + yiyj * g_13;
            double g_16 = weight0 * fac;
            double g_17 = c00z * g_16;
            double g_18 = g_16 * (c00z + zizj);
            double g_19 = b10 * g_16 + c00z * g_17 + zizj * g_17;
            double g_20 = c0pz * g_16;
            double g_21 = b00 * g_16 + c0pz * g_17;
            double g_22 = b00 * g_16 + c0pz * g_17 + zizj * g_20;
            double g_23 = b00 * g_17 + b10 * g_20 + c00z * g_21 + zizj * g_21;
            gout0 += g_7 * g_8 * g_16;
            gout1 += g_6 * g_9 * g_16;
            gout2 += g_6 * g_8 * g_17;
            gout3 += g_5 * g_10 * g_16;
            gout4 += g_4 * g_11 * g_16;
            gout5 += g_4 * g_10 * g_17;
            gout6 += g_5 * g_8 * g_18;
            gout7 += g_4 * g_9 * g_18;
            gout8 += g_4 * g_8 * g_19;
            gout9 += g_3 * g_12 * g_16;
            gout10 += g_2 * g_13 * g_16;
            gout11 += g_2 * g_12 * g_17;
            gout12 += g_1 * g_14 * g_16;
            gout13 += g_0 * g_15 * g_16;
            gout14 += g_0 * g_14 * g_17;
            gout15 += g_1 * g_12 * g_18;
            gout16 += g_0 * g_13 * g_18;
            gout17 += g_0 * g_12 * g_19;
            gout18 += g_3 * g_8 * g_20;
            gout19 += g_2 * g_9 * g_20;
            gout20 += g_2 * g_8 * g_21;
            gout21 += g_1 * g_10 * g_20;
            gout22 += g_0 * g_11 * g_20;
            gout23 += g_0 * g_10 * g_21;
            gout24 += g_1 * g_8 * g_22;
            gout25 += g_0 * g_9 * g_22;
            gout26 += g_0 * g_8 * g_23;
        }
    } }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            d_3 = dm[(i0+0)+nao*(j0+1)];
            d_4 = dm[(i0+1)+nao*(j0+1)];
            d_5 = dm[(i0+2)+nao*(j0+1)];
            d_6 = dm[(i0+0)+nao*(j0+2)];
            d_7 = dm[(i0+1)+nao*(j0+2)];
            d_8 = dm[(i0+2)+nao*(j0+2)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8);
            atomicAdd(vj+(k0+1)+nao*(l0+0), gout9*d_0 + gout10*d_1 + gout11*d_2 + gout12*d_3 + gout13*d_4 + gout14*d_5 + gout15*d_6 + gout16*d_7 + gout17*d_8);
            atomicAdd(vj+(k0+2)+nao*(l0+0), gout18*d_0 + gout19*d_1 + gout20*d_2 + gout21*d_3 + gout22*d_4 + gout23*d_5 + gout24*d_6 + gout25*d_7 + gout26*d_8);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+1)+nao*(l0+0)];
            d_2 = dm[(k0+2)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0 + gout9*d_1 + gout18*d_2);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0 + gout10*d_1 + gout19*d_2);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0 + gout11*d_1 + gout20*d_2);
            atomicAdd(vj+(i0+0)+nao*(j0+1), gout3*d_0 + gout12*d_1 + gout21*d_2);
            atomicAdd(vj+(i0+1)+nao*(j0+1), gout4*d_0 + gout13*d_1 + gout22*d_2);
            atomicAdd(vj+(i0+2)+nao*(j0+1), gout5*d_0 + gout14*d_1 + gout23*d_2);
            atomicAdd(vj+(i0+0)+nao*(j0+2), gout6*d_0 + gout15*d_1 + gout24*d_2);
            atomicAdd(vj+(i0+1)+nao*(j0+2), gout7*d_0 + gout16*d_1 + gout25*d_2);
            atomicAdd(vj+(i0+2)+nao*(j0+2), gout8*d_0 + gout17*d_1 + gout26*d_2);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            d_1 = dm[(j0+1)+nao*(l0+0)];
            d_2 = dm[(j0+2)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0 + gout3*d_1 + gout6*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0 + gout4*d_1 + gout7*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0 + gout5*d_1 + gout8*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout9*d_0 + gout12*d_1 + gout15*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+1), gout10*d_0 + gout13*d_1 + gout16*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+1), gout11*d_0 + gout14*d_1 + gout17*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout18*d_0 + gout21*d_1 + gout24*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+2), gout19*d_0 + gout22*d_1 + gout25*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+2), gout20*d_0 + gout23*d_1 + gout26*d_2);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+1)+nao*(k0+0)];
            d_2 = dm[(j0+2)+nao*(k0+0)];
            d_3 = dm[(j0+0)+nao*(k0+1)];
            d_4 = dm[(j0+1)+nao*(k0+1)];
            d_5 = dm[(j0+2)+nao*(k0+1)];
            d_6 = dm[(j0+0)+nao*(k0+2)];
            d_7 = dm[(j0+1)+nao*(k0+2)];
            d_8 = dm[(j0+2)+nao*(k0+2)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0 + gout3*d_1 + gout6*d_2 + gout9*d_3 + gout12*d_4 + gout15*d_5 + gout18*d_6 + gout21*d_7 + gout24*d_8);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0 + gout4*d_1 + gout7*d_2 + gout10*d_3 + gout13*d_4 + gout16*d_5 + gout19*d_6 + gout22*d_7 + gout25*d_8);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0 + gout5*d_1 + gout8*d_2 + gout11*d_3 + gout14*d_4 + gout17*d_5 + gout20*d_6 + gout23*d_7 + gout26*d_8);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2);
            atomicAdd(vk+(j0+1)+nao*(k0+0), gout3*d_0 + gout4*d_1 + gout5*d_2);
            atomicAdd(vk+(j0+2)+nao*(k0+0), gout6*d_0 + gout7*d_1 + gout8*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout9*d_0 + gout10*d_1 + gout11*d_2);
            atomicAdd(vk+(j0+1)+nao*(k0+1), gout12*d_0 + gout13*d_1 + gout14*d_2);
            atomicAdd(vk+(j0+2)+nao*(k0+1), gout15*d_0 + gout16*d_1 + gout17*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout18*d_0 + gout19*d_1 + gout20*d_2);
            atomicAdd(vk+(j0+1)+nao*(k0+2), gout21*d_0 + gout22*d_1 + gout23*d_2);
            atomicAdd(vk+(j0+2)+nao*(k0+2), gout24*d_0 + gout25*d_1 + gout26*d_2);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+0)+nao*(k0+1)];
            d_4 = dm[(i0+1)+nao*(k0+1)];
            d_5 = dm[(i0+2)+nao*(k0+1)];
            d_6 = dm[(i0+0)+nao*(k0+2)];
            d_7 = dm[(i0+1)+nao*(k0+2)];
            d_8 = dm[(i0+2)+nao*(k0+2)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout9*d_3 + gout10*d_4 + gout11*d_5 + gout18*d_6 + gout19*d_7 + gout20*d_8);
            atomicAdd(vk+(j0+1)+nao*(l0+0), gout3*d_0 + gout4*d_1 + gout5*d_2 + gout12*d_3 + gout13*d_4 + gout14*d_5 + gout21*d_6 + gout22*d_7 + gout23*d_8);
            atomicAdd(vk+(j0+2)+nao*(l0+0), gout6*d_0 + gout7*d_1 + gout8*d_2 + gout15*d_3 + gout16*d_4 + gout17*d_5 + gout24*d_6 + gout25*d_7 + gout26*d_8);
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_jk_kernel2000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double b10 = b00 + tmp4 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c00x * c00x + b10;
            double g_3 = 1;
            double g_4 = c00y;
            double g_5 = c00y * c00y + b10;
            double g_6 = weight0 * fac;
            double g_7 = c00z * g_6;
            double g_8 = b10 * g_6 + c00z * g_7;
            gout0 += g_2 * g_3 * g_6;
            gout1 += g_1 * g_4 * g_6;
            gout2 += g_1 * g_3 * g_7;
            gout3 += g_0 * g_5 * g_6;
            gout4 += g_0 * g_4 * g_7;
            gout5 += g_0 * g_3 * g_8;
        }
    } }
    double d_0, d_1, d_2, d_3, d_4, d_5;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            d_3 = dm[(i0+3)+nao*(j0+0)];
            d_4 = dm[(i0+4)+nao*(j0+0)];
            d_5 = dm[(i0+5)+nao*(j0+0)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+0), gout3*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+0), gout4*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+0), gout5*d_0);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+0), gout3*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+0), gout4*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+0), gout5*d_0);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0);
            atomicAdd(vk+(i0+3)+nao*(l0+0), gout3*d_0);
            atomicAdd(vk+(i0+4)+nao*(l0+0), gout4*d_0);
            atomicAdd(vk+(i0+5)+nao*(l0+0), gout5*d_0);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            d_3 = dm[(i0+3)+nao*(l0+0)];
            d_4 = dm[(i0+4)+nao*(l0+0)];
            d_5 = dm[(i0+5)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+3)+nao*(k0+0)];
            d_4 = dm[(i0+4)+nao*(k0+0)];
            d_5 = dm[(i0+5)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5);
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_jk_kernel2010(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double b10 = b00 + tmp4 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double tmp3 = tmp1 * aij;
            double c0px = xkl - xk + tmp3 * xijxkl;
            double c0py = ykl - yk + tmp3 * yijykl;
            double c0pz = zkl - zk + tmp3 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c00x * c00x + b10;
            double g_3 = c0px;
            double g_4 = c0px * c00x + b00;
            double g_5 = b00 * c00x + b10 * c0px + c00x * g_4;
            double g_6 = 1;
            double g_7 = c00y;
            double g_8 = c00y * c00y + b10;
            double g_9 = c0py;
            double g_10 = c0py * c00y + b00;
            double g_11 = b00 * c00y + b10 * c0py + c00y * g_10;
            double g_12 = weight0 * fac;
            double g_13 = c00z * g_12;
            double g_14 = b10 * g_12 + c00z * g_13;
            double g_15 = c0pz * g_12;
            double g_16 = b00 * g_12 + c0pz * g_13;
            double g_17 = b00 * g_13 + b10 * g_15 + c00z * g_16;
            gout0 += g_5 * g_6 * g_12;
            gout1 += g_4 * g_7 * g_12;
            gout2 += g_4 * g_6 * g_13;
            gout3 += g_3 * g_8 * g_12;
            gout4 += g_3 * g_7 * g_13;
            gout5 += g_3 * g_6 * g_14;
            gout6 += g_2 * g_9 * g_12;
            gout7 += g_1 * g_10 * g_12;
            gout8 += g_1 * g_9 * g_13;
            gout9 += g_0 * g_11 * g_12;
            gout10 += g_0 * g_10 * g_13;
            gout11 += g_0 * g_9 * g_14;
            gout12 += g_2 * g_6 * g_15;
            gout13 += g_1 * g_7 * g_15;
            gout14 += g_1 * g_6 * g_16;
            gout15 += g_0 * g_8 * g_15;
            gout16 += g_0 * g_7 * g_16;
            gout17 += g_0 * g_6 * g_17;
        }
    } }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            d_3 = dm[(i0+3)+nao*(j0+0)];
            d_4 = dm[(i0+4)+nao*(j0+0)];
            d_5 = dm[(i0+5)+nao*(j0+0)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5);
            atomicAdd(vj+(k0+1)+nao*(l0+0), gout6*d_0 + gout7*d_1 + gout8*d_2 + gout9*d_3 + gout10*d_4 + gout11*d_5);
            atomicAdd(vj+(k0+2)+nao*(l0+0), gout12*d_0 + gout13*d_1 + gout14*d_2 + gout15*d_3 + gout16*d_4 + gout17*d_5);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+1)+nao*(l0+0)];
            d_2 = dm[(k0+2)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0 + gout6*d_1 + gout12*d_2);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0 + gout7*d_1 + gout13*d_2);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0 + gout8*d_1 + gout14*d_2);
            atomicAdd(vj+(i0+3)+nao*(j0+0), gout3*d_0 + gout9*d_1 + gout15*d_2);
            atomicAdd(vj+(i0+4)+nao*(j0+0), gout4*d_0 + gout10*d_1 + gout16*d_2);
            atomicAdd(vj+(i0+5)+nao*(j0+0), gout5*d_0 + gout11*d_1 + gout17*d_2);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+0), gout3*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+0), gout4*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+0), gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout6*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+1), gout7*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+1), gout8*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+1), gout9*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+1), gout10*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+1), gout11*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout12*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+2), gout13*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+2), gout14*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+2), gout15*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+2), gout16*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+2), gout17*d_0);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+0)+nao*(k0+1)];
            d_2 = dm[(j0+0)+nao*(k0+2)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0 + gout6*d_1 + gout12*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0 + gout7*d_1 + gout13*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0 + gout8*d_1 + gout14*d_2);
            atomicAdd(vk+(i0+3)+nao*(l0+0), gout3*d_0 + gout9*d_1 + gout15*d_2);
            atomicAdd(vk+(i0+4)+nao*(l0+0), gout4*d_0 + gout10*d_1 + gout16*d_2);
            atomicAdd(vk+(i0+5)+nao*(l0+0), gout5*d_0 + gout11*d_1 + gout17*d_2);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            d_3 = dm[(i0+3)+nao*(l0+0)];
            d_4 = dm[(i0+4)+nao*(l0+0)];
            d_5 = dm[(i0+5)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout6*d_0 + gout7*d_1 + gout8*d_2 + gout9*d_3 + gout10*d_4 + gout11*d_5);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout12*d_0 + gout13*d_1 + gout14*d_2 + gout15*d_3 + gout16*d_4 + gout17*d_5);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+3)+nao*(k0+0)];
            d_4 = dm[(i0+4)+nao*(k0+0)];
            d_5 = dm[(i0+5)+nao*(k0+0)];
            d_6 = dm[(i0+0)+nao*(k0+1)];
            d_7 = dm[(i0+1)+nao*(k0+1)];
            d_8 = dm[(i0+2)+nao*(k0+1)];
            d_9 = dm[(i0+3)+nao*(k0+1)];
            d_10 = dm[(i0+4)+nao*(k0+1)];
            d_11 = dm[(i0+5)+nao*(k0+1)];
            d_12 = dm[(i0+0)+nao*(k0+2)];
            d_13 = dm[(i0+1)+nao*(k0+2)];
            d_14 = dm[(i0+2)+nao*(k0+2)];
            d_15 = dm[(i0+3)+nao*(k0+2)];
            d_16 = dm[(i0+4)+nao*(k0+2)];
            d_17 = dm[(i0+5)+nao*(k0+2)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8 + gout9*d_9 + gout10*d_10 + gout11*d_11 + gout12*d_12 + gout13*d_13 + gout14*d_14 + gout15*d_15 + gout16*d_16 + gout17*d_17);
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_jk_kernel2100(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double b10 = b00 + tmp4 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c00x * c00x + b10;
            double g_3 = c00x + xixj;
            double g_4 = c00x * (c00x + xixj) + b10;
            double g_5 = c00x * (2 * b10 + g_2) + xixj * g_2;
            double g_6 = 1;
            double g_7 = c00y;
            double g_8 = c00y * c00y + b10;
            double g_9 = c00y + yiyj;
            double g_10 = c00y * (c00y + yiyj) + b10;
            double g_11 = c00y * (2 * b10 + g_8) + yiyj * g_8;
            double g_12 = weight0 * fac;
            double g_13 = c00z * g_12;
            double g_14 = b10 * g_12 + c00z * g_13;
            double g_15 = g_12 * (c00z + zizj);
            double g_16 = b10 * g_12 + c00z * g_13 + zizj * g_13;
            double g_17 = 2 * b10 * g_13 + c00z * g_14 + zizj * g_14;
            gout0 += g_5 * g_6 * g_12;
            gout1 += g_4 * g_7 * g_12;
            gout2 += g_4 * g_6 * g_13;
            gout3 += g_3 * g_8 * g_12;
            gout4 += g_3 * g_7 * g_13;
            gout5 += g_3 * g_6 * g_14;
            gout6 += g_2 * g_9 * g_12;
            gout7 += g_1 * g_10 * g_12;
            gout8 += g_1 * g_9 * g_13;
            gout9 += g_0 * g_11 * g_12;
            gout10 += g_0 * g_10 * g_13;
            gout11 += g_0 * g_9 * g_14;
            gout12 += g_2 * g_6 * g_15;
            gout13 += g_1 * g_7 * g_15;
            gout14 += g_1 * g_6 * g_16;
            gout15 += g_0 * g_8 * g_15;
            gout16 += g_0 * g_7 * g_16;
            gout17 += g_0 * g_6 * g_17;
        }
    } }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            d_3 = dm[(i0+3)+nao*(j0+0)];
            d_4 = dm[(i0+4)+nao*(j0+0)];
            d_5 = dm[(i0+5)+nao*(j0+0)];
            d_6 = dm[(i0+0)+nao*(j0+1)];
            d_7 = dm[(i0+1)+nao*(j0+1)];
            d_8 = dm[(i0+2)+nao*(j0+1)];
            d_9 = dm[(i0+3)+nao*(j0+1)];
            d_10 = dm[(i0+4)+nao*(j0+1)];
            d_11 = dm[(i0+5)+nao*(j0+1)];
            d_12 = dm[(i0+0)+nao*(j0+2)];
            d_13 = dm[(i0+1)+nao*(j0+2)];
            d_14 = dm[(i0+2)+nao*(j0+2)];
            d_15 = dm[(i0+3)+nao*(j0+2)];
            d_16 = dm[(i0+4)+nao*(j0+2)];
            d_17 = dm[(i0+5)+nao*(j0+2)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8 + gout9*d_9 + gout10*d_10 + gout11*d_11 + gout12*d_12 + gout13*d_13 + gout14*d_14 + gout15*d_15 + gout16*d_16 + gout17*d_17);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+0), gout3*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+0), gout4*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+0), gout5*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+1), gout6*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+1), gout7*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+1), gout8*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+1), gout9*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+1), gout10*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+1), gout11*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+2), gout12*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+2), gout13*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+2), gout14*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+2), gout15*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+2), gout16*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+2), gout17*d_0);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            d_1 = dm[(j0+1)+nao*(l0+0)];
            d_2 = dm[(j0+2)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0 + gout6*d_1 + gout12*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0 + gout7*d_1 + gout13*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0 + gout8*d_1 + gout14*d_2);
            atomicAdd(vk+(i0+3)+nao*(k0+0), gout3*d_0 + gout9*d_1 + gout15*d_2);
            atomicAdd(vk+(i0+4)+nao*(k0+0), gout4*d_0 + gout10*d_1 + gout16*d_2);
            atomicAdd(vk+(i0+5)+nao*(k0+0), gout5*d_0 + gout11*d_1 + gout17*d_2);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+1)+nao*(k0+0)];
            d_2 = dm[(j0+2)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0 + gout6*d_1 + gout12*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0 + gout7*d_1 + gout13*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0 + gout8*d_1 + gout14*d_2);
            atomicAdd(vk+(i0+3)+nao*(l0+0), gout3*d_0 + gout9*d_1 + gout15*d_2);
            atomicAdd(vk+(i0+4)+nao*(l0+0), gout4*d_0 + gout10*d_1 + gout16*d_2);
            atomicAdd(vk+(i0+5)+nao*(l0+0), gout5*d_0 + gout11*d_1 + gout17*d_2);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            d_3 = dm[(i0+3)+nao*(l0+0)];
            d_4 = dm[(i0+4)+nao*(l0+0)];
            d_5 = dm[(i0+5)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5);
            atomicAdd(vk+(j0+1)+nao*(k0+0), gout6*d_0 + gout7*d_1 + gout8*d_2 + gout9*d_3 + gout10*d_4 + gout11*d_5);
            atomicAdd(vk+(j0+2)+nao*(k0+0), gout12*d_0 + gout13*d_1 + gout14*d_2 + gout15*d_3 + gout16*d_4 + gout17*d_5);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+3)+nao*(k0+0)];
            d_4 = dm[(i0+4)+nao*(k0+0)];
            d_5 = dm[(i0+5)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5);
            atomicAdd(vk+(j0+1)+nao*(l0+0), gout6*d_0 + gout7*d_1 + gout8*d_2 + gout9*d_3 + gout10*d_4 + gout11*d_5);
            atomicAdd(vk+(j0+2)+nao*(l0+0), gout12*d_0 + gout13*d_1 + gout14*d_2 + gout15*d_3 + gout16*d_4 + gout17*d_5);
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_jk_kernel3000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }
    double omega = envs.omega;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = c_bpcache.ao_loc;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    int ij, kl, i_dm;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        double rw[4];
        double root0, weight0;
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        int irys;
        for (irys = 0; irys < 2; ++irys) {
            root0 = rw[irys];
            weight0 = rw[irys+2];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double b00 = u2 * tmp4;
            double tmp1 = 2 * b00;
            double tmp2 = tmp1 * akl;
            double b10 = b00 + tmp4 * akl;
            double c00x = xij - xi - tmp2 * xijxkl;
            double c00y = yij - yi - tmp2 * yijykl;
            double c00z = zij - zi - tmp2 * zijzkl;
            double g_0 = 1;
            double g_1 = c00x;
            double g_2 = c00x * c00x + b10;
            double g_3 = c00x * (2 * b10 + g_2);
            double g_4 = 1;
            double g_5 = c00y;
            double g_6 = c00y * c00y + b10;
            double g_7 = c00y * (2 * b10 + g_6);
            double g_8 = weight0 * fac;
            double g_9 = c00z * g_8;
            double g_10 = b10 * g_8 + c00z * g_9;
            double g_11 = 2 * b10 * g_9 + c00z * g_10;
            gout0 += g_3 * g_4 * g_8;
            gout1 += g_2 * g_5 * g_8;
            gout2 += g_2 * g_4 * g_9;
            gout3 += g_1 * g_6 * g_8;
            gout4 += g_1 * g_5 * g_9;
            gout5 += g_1 * g_4 * g_10;
            gout6 += g_0 * g_7 * g_8;
            gout7 += g_0 * g_6 * g_9;
            gout8 += g_0 * g_5 * g_10;
            gout9 += g_0 * g_4 * g_11;
        }
    } }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0+0)+nao*(j0+0)];
            d_1 = dm[(i0+1)+nao*(j0+0)];
            d_2 = dm[(i0+2)+nao*(j0+0)];
            d_3 = dm[(i0+3)+nao*(j0+0)];
            d_4 = dm[(i0+4)+nao*(j0+0)];
            d_5 = dm[(i0+5)+nao*(j0+0)];
            d_6 = dm[(i0+6)+nao*(j0+0)];
            d_7 = dm[(i0+7)+nao*(j0+0)];
            d_8 = dm[(i0+8)+nao*(j0+0)];
            d_9 = dm[(i0+9)+nao*(j0+0)];
            atomicAdd(vj+(k0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8 + gout9*d_9);
            // ijkl,kl->ij
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+0), gout3*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+0), gout4*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+0), gout5*d_0);
            atomicAdd(vj+(i0+6)+nao*(j0+0), gout6*d_0);
            atomicAdd(vj+(i0+7)+nao*(j0+0), gout7*d_0);
            atomicAdd(vj+(i0+8)+nao*(j0+0), gout8*d_0);
            atomicAdd(vj+(i0+9)+nao*(j0+0), gout9*d_0);
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+0), gout3*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+0), gout4*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+0), gout5*d_0);
            atomicAdd(vk+(i0+6)+nao*(k0+0), gout6*d_0);
            atomicAdd(vk+(i0+7)+nao*(k0+0), gout7*d_0);
            atomicAdd(vk+(i0+8)+nao*(k0+0), gout8*d_0);
            atomicAdd(vk+(i0+9)+nao*(k0+0), gout9*d_0);
            // ijkl,jk->il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0);
            atomicAdd(vk+(i0+3)+nao*(l0+0), gout3*d_0);
            atomicAdd(vk+(i0+4)+nao*(l0+0), gout4*d_0);
            atomicAdd(vk+(i0+5)+nao*(l0+0), gout5*d_0);
            atomicAdd(vk+(i0+6)+nao*(l0+0), gout6*d_0);
            atomicAdd(vk+(i0+7)+nao*(l0+0), gout7*d_0);
            atomicAdd(vk+(i0+8)+nao*(l0+0), gout8*d_0);
            atomicAdd(vk+(i0+9)+nao*(l0+0), gout9*d_0);
            // ijkl,il->jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            d_3 = dm[(i0+3)+nao*(l0+0)];
            d_4 = dm[(i0+4)+nao*(l0+0)];
            d_5 = dm[(i0+5)+nao*(l0+0)];
            d_6 = dm[(i0+6)+nao*(l0+0)];
            d_7 = dm[(i0+7)+nao*(l0+0)];
            d_8 = dm[(i0+8)+nao*(l0+0)];
            d_9 = dm[(i0+9)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8 + gout9*d_9);
            // ijkl,ik->jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+3)+nao*(k0+0)];
            d_4 = dm[(i0+4)+nao*(k0+0)];
            d_5 = dm[(i0+5)+nao*(k0+0)];
            d_6 = dm[(i0+6)+nao*(k0+0)];
            d_7 = dm[(i0+7)+nao*(k0+0)];
            d_8 = dm[(i0+8)+nao*(k0+0)];
            d_9 = dm[(i0+9)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout0*d_0 + gout1*d_1 + gout2*d_2 + gout3*d_3 + gout4*d_4 + gout5*d_5 + gout6*d_6 + gout7*d_7 + gout8*d_8 + gout9*d_9);
            vk += nao2;
        }
        dm += nao2;
    }
}
