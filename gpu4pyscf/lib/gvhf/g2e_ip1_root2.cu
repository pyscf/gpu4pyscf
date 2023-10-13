/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2023 Qiming Sun
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
static void GINTint2e_ip1_jk_kernel_0010(GINTEnvVars envs, JKMatrix jk,
                                    BasisProdOffsets offsets)
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
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double * __restrict__ i_exponent = c_bpcache.a1;
    double * __restrict__ j_exponent = c_bpcache.a2;
    int ij, kl;
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
    double ABx = xi - bas_x[jsh];
    double ABy = yi - bas_y[jsh];
    double ABz = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    double rw[4];
    int irys;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
        double alpha = i_exponent[ij];
        double beta = j_exponent[ij];
		    double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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
        
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
            double weight0 = rw[irys+2] * fac;
            double root0 = rw[irys];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double B00 = u2 * tmp4;
            double tmp1 = 2 * B00;    
            
            double tmp2 = tmp1 * akl;
            double C00x = xij - xi - tmp2 * xijxkl;
            double C00y = yij - yi - tmp2 * yijykl;
            double C00z = zij - zi - tmp2 * zijzkl;
            
            double tmp3 = tmp1 * aij;
            double D00x = xkl - xk + tmp3 * xijxkl;
            double D00y = ykl - yk + tmp3 * yijykl;
            double D00z = zkl - zk + tmp3 * zijzkl;

            double g1 = B00+(C00x*D00x);
            double g2 = ABx+C00x;
            double g4 = B00+(g2*D00x);
            double g6 = B00+(C00y*D00y);
            double g7 = ABy+C00y;
            double g9 = B00+(g7*D00y);
            double g11 = B00+(C00z*D00z);
            double g12 = ABz+C00z;
            double g14 = B00+(g12*D00z);

            gout0 += (2*alpha*g1) * weight0;
            gout1 += (2*alpha*C00x) * (D00y) * weight0;
            gout2 += (2*alpha*C00x) * (D00z) * weight0;
            gout3 += (D00x) * (2*alpha*C00y) * weight0;
            gout4 += (2*alpha*g6) * weight0;
            gout5 += (2*alpha*C00y) * (D00z) * weight0;
            gout6 += (D00x) * (2*alpha*C00z) * weight0;
            gout7 += (D00y) * (2*alpha*C00z) * weight0;
            gout8 += (2*alpha*g11) * weight0;
            gout9 += (2*beta*g4) * weight0;
            gout10 += (2*beta*g2) * (D00y) * weight0;
            gout11 += (2*beta*g2) * (D00z) * weight0;
            gout12 += (D00x) * (2*beta*g7) * weight0;
            gout13 += (2*beta*g9) * weight0;
            gout14 += (2*beta*g7) * (D00z) * weight0;
            gout15 += (D00x) * (2*beta*g12) * weight0;
            gout16 += (D00y) * (2*beta*g12) * weight0;
            gout17 += (2*beta*g14) * weight0;
        }
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double d_0, d_1, d_2;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+1)+nao*(l0+0)];
            d_2 = dm[(k0+2)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0+gout1*d_1+gout2*d_2);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+nao2, gout3*d_0+gout4*d_1+gout5*d_2);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+2*nao2, gout6*d_0+gout7*d_1+gout8*d_2);
            atomicAdd(vj+nao*(i0+0)+(j0+0), gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+2*nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            vj += 3*nao2;
        }
        if (vk != NULL) {
            // ijkl, jk -> il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+0)+nao*(k0+1)];
            d_2 = dm[(j0+0)+nao*(k0+2)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0+gout1*d_1+gout2*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+nao2, gout3*d_0+gout4*d_1+gout5*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+2*nao2, gout6*d_0+gout7*d_1+gout8*d_2);
            // ijkl, jl -> ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout1*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout2*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+nao2, gout3*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+nao2, gout4*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+nao2, gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+2*nao2, gout6*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+2*nao2, gout7*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+2*nao2, gout8*d_0);
            // ijkl, ik -> jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+0)+nao*(k0+1)];
            d_2 = dm[(i0+0)+nao*(k0+2)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+2*nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            // ijkl, il -> jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout9*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout10*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout11*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+nao2, gout12*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+nao2, gout13*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+nao2, gout14*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+2*nao2, gout15*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+2*nao2, gout16*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+2*nao2, gout17*d_0);
            vk += 3*nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_ip1_jk_kernel_0011(GINTEnvVars envs, JKMatrix jk,
                                    BasisProdOffsets offsets)
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
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double * __restrict__ i_exponent = c_bpcache.a1;
    double * __restrict__ j_exponent = c_bpcache.a2;
    int ij, kl;
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
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;

    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double ABx = xi - bas_x[jsh];
    double ABy = yi - bas_y[jsh];
    double ABz = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double CDx = xk - bas_x[lsh];
    double CDy = yk - bas_y[lsh];
    double CDz = zk - bas_z[lsh];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    double rw[4];
    int irys;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
        double alpha = i_exponent[ij];
        double beta = j_exponent[ij];
		    double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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

        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
            double weight0 = rw[irys+2] * fac;
            double root0 = rw[irys];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double B00 = u2 * tmp4;
            double tmp1 = 2 * B00;    
            
            double tmp2 = tmp1 * akl;
            double C00x = xij - xi - tmp2 * xijxkl;
            double C00y = yij - yi - tmp2 * yijykl;
            double C00z = zij - zi - tmp2 * zijzkl;
            double B01 = B00 + tmp4 * aij;
            double tmp3 = tmp1 * aij;
            double D00x = xkl - xk + tmp3 * xijxkl;
            double D00y = ykl - yk + tmp3 * yijykl;
            double D00z = zkl - zk + tmp3 * zijzkl;

            double g1 = B00+(C00x*D00x);
            double g2 = ABx+C00x;
            double g3 = g2*D00x;
            double g4 = B00+g3;
            double g5 = CDx+D00x;
            double g7 = B00+(C00x*g5);
            double g9 = B01+(D00x*g5);
            double g14 = B00*(CDx+(2*D00x));
            double g15 = (B01*C00x)+(C00x*D00x*g5)+g14;
            double g17 = B00+(g2*g5);
            double g19 = g14+(g2*g9);
            double g21 = B00+(C00y*D00y);
            double g22 = ABy+C00y;
            double g24 = B00+(g22*D00y);
            double g25 = CDy+D00y;
            double g27 = B00+(C00y*g25);
            double g29 = B01+(D00y*g25);
            double g34 = B00*(CDy+(2*D00y));
            double g35 = (B01*C00y)+(C00y*D00y*g25)+g34;
            double g37 = B00+(g22*g25);
            double g39 = g34+(g22*g29);
            double g41 = B00+(C00z*D00z);
            double g42 = ABz+C00z;
            double g44 = B00+(g42*D00z);
            double g45 = CDz+D00z;
            double g47 = B00+(C00z*g45);
            double g49 = B01+(D00z*g45);
            double g54 = B00*(CDz+(2*D00z));
            double g55 = (B01*C00z)+(C00z*D00z*g45)+g54;
            double g57 = B00+(g42*g45);
            double g59 = g54+(g42*g49);

            gout0 += (2*alpha*g15) * weight0;
            gout1 += (2*alpha*g1) * (g25) * weight0;
            gout2 += (2*alpha*g1) * (g45) * weight0;
            gout3 += (2*alpha*g7) * (D00y) * weight0;
            gout4 += (2*alpha*C00x) * (g29) * weight0;
            gout5 += (2*alpha*C00x) * (D00y) * (g45) * weight0;
            gout6 += (2*alpha*g7) * (D00z) * weight0;
            gout7 += (2*alpha*C00x) * (g25) * (D00z) * weight0;
            gout8 += (2*alpha*C00x) * (g49) * weight0;
            gout9 += (g9) * (2*alpha*C00y) * weight0;
            gout10 += (D00x) * (2*alpha*g27) * weight0;
            gout11 += (D00x) * (2*alpha*C00y) * (g45) * weight0;
            gout12 += (g5) * (2*alpha*g21) * weight0;
            gout13 += (2*alpha*g35) * weight0;
            gout14 += (2*alpha*g21) * (g45) * weight0;
            gout15 += (g5) * (2*alpha*C00y) * (D00z) * weight0;
            gout16 += (2*alpha*g27) * (D00z) * weight0;
            gout17 += (2*alpha*C00y) * (g49) * weight0;
            gout18 += (g9) * (2*alpha*C00z) * weight0;
            gout19 += (D00x) * (g25) * (2*alpha*C00z) * weight0;
            gout20 += (D00x) * (2*alpha*g47) * weight0;
            gout21 += (g5) * (D00y) * (2*alpha*C00z) * weight0;
            gout22 += (g29) * (2*alpha*C00z) * weight0;
            gout23 += (D00y) * (2*alpha*g47) * weight0;
            gout24 += (g5) * (2*alpha*g41) * weight0;
            gout25 += (g25) * (2*alpha*g41) * weight0;
            gout26 += (2*alpha*g55) * weight0;
            gout27 += (2*beta*g19) * weight0;
            gout28 += (2*beta*g4) * (g25) * weight0;
            gout29 += (2*beta*g4) * (g45) * weight0;
            gout30 += (2*beta*g17) * (D00y) * weight0;
            gout31 += (2*beta*g2) * (g29) * weight0;
            gout32 += (2*beta*g2) * (D00y) * (g45) * weight0;
            gout33 += (2*beta*g17) * (D00z) * weight0;
            gout34 += (2*beta*g2) * (g25) * (D00z) * weight0;
            gout35 += (2*beta*g2) * (g49) * weight0;
            gout36 += (g9) * (2*beta*g22) * weight0;
            gout37 += (D00x) * (2*beta*g37) * weight0;
            gout38 += (D00x) * (2*beta*g22) * (g45) * weight0;
            gout39 += (g5) * (2*beta*g24) * weight0;
            gout40 += (2*beta*g39) * weight0;
            gout41 += (2*beta*g24) * (g45) * weight0;
            gout42 += (g5) * (2*beta*g22) * (D00z) * weight0;
            gout43 += (2*beta*g37) * (D00z) * weight0;
            gout44 += (2*beta*g22) * (g49) * weight0;
            gout45 += (g9) * (2*beta*g42) * weight0;
            gout46 += (D00x) * (g25) * (2*beta*g42) * weight0;
            gout47 += (D00x) * (2*beta*g57) * weight0;
            gout48 += (g5) * (D00y) * (2*beta*g42) * weight0;
            gout49 += (g29) * (2*beta*g42) * weight0;
            gout50 += (D00y) * (2*beta*g57) * weight0;
            gout51 += (g5) * (2*beta*g44) * weight0;
            gout52 += (g25) * (2*beta*g44) * weight0;
            gout53 += (2*beta*g59) * weight0;
        }
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+0)+nao*(l0+1)];
            d_2 = dm[(k0+0)+nao*(l0+2)];
            d_3 = dm[(k0+1)+nao*(l0+0)];
            d_4 = dm[(k0+1)+nao*(l0+1)];
            d_5 = dm[(k0+1)+nao*(l0+2)];
            d_6 = dm[(k0+2)+nao*(l0+0)];
            d_7 = dm[(k0+2)+nao*(l0+1)];
            d_8 = dm[(k0+2)+nao*(l0+2)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0+gout1*d_1+gout2*d_2+gout3*d_3+gout4*d_4+gout5*d_5+gout6*d_6+gout7*d_7+gout8*d_8);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+nao2, gout9*d_0+gout10*d_1+gout11*d_2+gout12*d_3+gout13*d_4+gout14*d_5+gout15*d_6+gout16*d_7+gout17*d_8);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+2*nao2, gout18*d_0+gout19*d_1+gout20*d_2+gout21*d_3+gout22*d_4+gout23*d_5+gout24*d_6+gout25*d_7+gout26*d_8);
            atomicAdd(vj+nao*(i0+0)+(j0+0), gout27*d_0+gout28*d_1+gout29*d_2+gout30*d_3+gout31*d_4+gout32*d_5+gout33*d_6+gout34*d_7+gout35*d_8);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+nao2, gout36*d_0+gout37*d_1+gout38*d_2+gout39*d_3+gout40*d_4+gout41*d_5+gout42*d_6+gout43*d_7+gout44*d_8);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+2*nao2, gout45*d_0+gout46*d_1+gout47*d_2+gout48*d_3+gout49*d_4+gout50*d_5+gout51*d_6+gout52*d_7+gout53*d_8);
            vj += 3*nao2;
        }
        if (vk != NULL) {
            // ijkl, jk -> il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+0)+nao*(k0+1)];
            d_2 = dm[(j0+0)+nao*(k0+2)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0+gout3*d_1+gout6*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+1), gout1*d_0+gout4*d_1+gout7*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+2), gout2*d_0+gout5*d_1+gout8*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+nao2, gout9*d_0+gout12*d_1+gout15*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+1)+nao2, gout10*d_0+gout13*d_1+gout16*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+2)+nao2, gout11*d_0+gout14*d_1+gout17*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+2*nao2, gout18*d_0+gout21*d_1+gout24*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+1)+2*nao2, gout19*d_0+gout22*d_1+gout25*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+2)+2*nao2, gout20*d_0+gout23*d_1+gout26*d_2);
            // ijkl, jl -> ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            d_1 = dm[(j0+0)+nao*(l0+1)];
            d_2 = dm[(j0+0)+nao*(l0+2)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0+gout1*d_1+gout2*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout3*d_0+gout4*d_1+gout5*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout6*d_0+gout7*d_1+gout8*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+nao2, gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+2*nao2, gout18*d_0+gout19*d_1+gout20*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+2*nao2, gout21*d_0+gout22*d_1+gout23*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+2*nao2, gout24*d_0+gout25*d_1+gout26*d_2);
            // ijkl, ik -> jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+0)+nao*(k0+1)];
            d_2 = dm[(i0+0)+nao*(k0+2)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout27*d_0+gout30*d_1+gout33*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+1), gout28*d_0+gout31*d_1+gout34*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+2), gout29*d_0+gout32*d_1+gout35*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+nao2, gout36*d_0+gout39*d_1+gout42*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+1)+nao2, gout37*d_0+gout40*d_1+gout43*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+2)+nao2, gout38*d_0+gout41*d_1+gout44*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+2*nao2, gout45*d_0+gout48*d_1+gout51*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+1)+2*nao2, gout46*d_0+gout49*d_1+gout52*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+2)+2*nao2, gout47*d_0+gout50*d_1+gout53*d_2);
            // ijkl, il -> jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+0)+nao*(l0+1)];
            d_2 = dm[(i0+0)+nao*(l0+2)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout27*d_0+gout28*d_1+gout29*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout30*d_0+gout31*d_1+gout32*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout33*d_0+gout34*d_1+gout35*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+nao2, gout36*d_0+gout37*d_1+gout38*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+nao2, gout39*d_0+gout40*d_1+gout41*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+nao2, gout42*d_0+gout43*d_1+gout44*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+2*nao2, gout45*d_0+gout46*d_1+gout47*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+2*nao2, gout48*d_0+gout49*d_1+gout50*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+2*nao2, gout51*d_0+gout52*d_1+gout53*d_2);
            vk += 3*nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_ip1_jk_kernel_0020(GINTEnvVars envs, JKMatrix jk,
                                    BasisProdOffsets offsets)
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
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double * __restrict__ i_exponent = c_bpcache.a1;
    double * __restrict__ j_exponent = c_bpcache.a2;
    int ij, kl;
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
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;

    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double ABx = xi - bas_x[jsh];
    double ABy = yi - bas_y[jsh];
    double ABz = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    double rw[4];
    int irys;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
        double alpha = i_exponent[ij];
        double beta = j_exponent[ij];
		    double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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

        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
            double weight0 = rw[irys+2] * fac;
            double root0 = rw[irys];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double B00 = u2 * tmp4;
            double tmp1 = 2 * B00;    
            
            double tmp2 = tmp1 * akl;
            double C00x = xij - xi - tmp2 * xijxkl;
            double C00y = yij - yi - tmp2 * yijykl;
            double C00z = zij - zi - tmp2 * zijzkl;
            double B01 = B00 + tmp4 * aij;
            double tmp3 = tmp1 * aij;
            double D00x = xkl - xk + tmp3 * xijxkl;
            double D00y = ykl - yk + tmp3 * yijykl;
            double D00z = zkl - zk + tmp3 * zijzkl;

            double g1 = B00+(C00x*D00x);
            double g2 = B01+D00x*D00x;
            double g3 = 2*B00*D00x;
            double g5 = g3+(C00x*g2);
            double g6 = ABx+C00x;
            double g8 = B00+(g6*D00x);
            double g10 = g3+(g6*g2);
            double g12 = B00+(C00y*D00y);
            double g13 = B01+D00y*D00y;
            double g14 = 2*B00*D00y;
            double g16 = g14+(C00y*g13);
            double g17 = ABy+C00y;
            double g19 = B00+(g17*D00y);
            double g21 = g14+(g17*g13);
            double g23 = B00+(C00z*D00z);
            double g24 = B01+D00z*D00z;
            double g25 = 2*B00*D00z;
            double g27 = g25+(C00z*g24);
            double g28 = ABz+C00z;
            double g30 = B00+(g28*D00z);
            double g32 = g25+(g28*g24);

            gout0 += (2*alpha*g5) * weight0;
            gout1 += (2*alpha*g1) * (D00y) * weight0;
            gout2 += (2*alpha*g1) * (D00z) * weight0;
            gout3 += (2*alpha*C00x) * (g13) * weight0;
            gout4 += (2*alpha*C00x) * (D00y) * (D00z) * weight0;
            gout5 += (2*alpha*C00x) * (g24) * weight0;
            gout6 += (g2) * (2*alpha*C00y) * weight0;
            gout7 += (D00x) * (2*alpha*g12) * weight0;
            gout8 += (D00x) * (2*alpha*C00y) * (D00z) * weight0;
            gout9 += (2*alpha*g16) * weight0;
            gout10 += (2*alpha*g12) * (D00z) * weight0;
            gout11 += (2*alpha*C00y) * (g24) * weight0;
            gout12 += (g2) * (2*alpha*C00z) * weight0;
            gout13 += (D00x) * (D00y) * (2*alpha*C00z) * weight0;
            gout14 += (D00x) * (2*alpha*g23) * weight0;
            gout15 += (g13) * (2*alpha*C00z) * weight0;
            gout16 += (D00y) * (2*alpha*g23) * weight0;
            gout17 += (2*alpha*g27) * weight0;
            gout18 += (2*beta*g10) * weight0;
            gout19 += (2*beta*g8) * (D00y) * weight0;
            gout20 += (2*beta*g8) * (D00z) * weight0;
            gout21 += (2*beta*g6) * (g13) * weight0;
            gout22 += (2*beta*g6) * (D00y) * (D00z) * weight0;
            gout23 += (2*beta*g6) * (g24) * weight0;
            gout24 += (g2) * (2*beta*g17) * weight0;
            gout25 += (D00x) * (2*beta*g19) * weight0;
            gout26 += (D00x) * (2*beta*g17) * (D00z) * weight0;
            gout27 += (2*beta*g21) * weight0;
            gout28 += (2*beta*g19) * (D00z) * weight0;
            gout29 += (2*beta*g17) * (g24) * weight0;
            gout30 += (g2) * (2*beta*g28) * weight0;
            gout31 += (D00x) * (D00y) * (2*beta*g28) * weight0;
            gout32 += (D00x) * (2*beta*g30) * weight0;
            gout33 += (g13) * (2*beta*g28) * weight0;
            gout34 += (D00y) * (2*beta*g30) * weight0;
            gout35 += (2*beta*g32) * weight0;
        }
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double d_0, d_1, d_2, d_3, d_4, d_5;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+1)+nao*(l0+0)];
            d_2 = dm[(k0+2)+nao*(l0+0)];
            d_3 = dm[(k0+3)+nao*(l0+0)];
            d_4 = dm[(k0+4)+nao*(l0+0)];
            d_5 = dm[(k0+5)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0+gout1*d_1+gout2*d_2+gout3*d_3+gout4*d_4+gout5*d_5);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+nao2, gout6*d_0+gout7*d_1+gout8*d_2+gout9*d_3+gout10*d_4+gout11*d_5);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+2*nao2, gout12*d_0+gout13*d_1+gout14*d_2+gout15*d_3+gout16*d_4+gout17*d_5);
            atomicAdd(vj+nao*(i0+0)+(j0+0), gout18*d_0+gout19*d_1+gout20*d_2+gout21*d_3+gout22*d_4+gout23*d_5);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+nao2, gout24*d_0+gout25*d_1+gout26*d_2+gout27*d_3+gout28*d_4+gout29*d_5);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+2*nao2, gout30*d_0+gout31*d_1+gout32*d_2+gout33*d_3+gout34*d_4+gout35*d_5);
            vj += 3*nao2;
        }
        if (vk != NULL) {
            // ijkl, jk -> il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+0)+nao*(k0+1)];
            d_2 = dm[(j0+0)+nao*(k0+2)];
            d_3 = dm[(j0+0)+nao*(k0+3)];
            d_4 = dm[(j0+0)+nao*(k0+4)];
            d_5 = dm[(j0+0)+nao*(k0+5)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0+gout1*d_1+gout2*d_2+gout3*d_3+gout4*d_4+gout5*d_5);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+nao2, gout6*d_0+gout7*d_1+gout8*d_2+gout9*d_3+gout10*d_4+gout11*d_5);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+2*nao2, gout12*d_0+gout13*d_1+gout14*d_2+gout15*d_3+gout16*d_4+gout17*d_5);
            // ijkl, jl -> ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout1*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout2*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+3), gout3*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+4), gout4*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+5), gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+nao2, gout6*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+nao2, gout7*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+nao2, gout8*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+3)+nao2, gout9*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+4)+nao2, gout10*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+5)+nao2, gout11*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+2*nao2, gout12*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+2*nao2, gout13*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+2*nao2, gout14*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+3)+2*nao2, gout15*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+4)+2*nao2, gout16*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+5)+2*nao2, gout17*d_0);
            // ijkl, ik -> jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+0)+nao*(k0+1)];
            d_2 = dm[(i0+0)+nao*(k0+2)];
            d_3 = dm[(i0+0)+nao*(k0+3)];
            d_4 = dm[(i0+0)+nao*(k0+4)];
            d_5 = dm[(i0+0)+nao*(k0+5)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout18*d_0+gout19*d_1+gout20*d_2+gout21*d_3+gout22*d_4+gout23*d_5);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+nao2, gout24*d_0+gout25*d_1+gout26*d_2+gout27*d_3+gout28*d_4+gout29*d_5);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+2*nao2, gout30*d_0+gout31*d_1+gout32*d_2+gout33*d_3+gout34*d_4+gout35*d_5);
            // ijkl, il -> jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout18*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout19*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout20*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+3), gout21*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+4), gout22*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+5), gout23*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+nao2, gout24*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+nao2, gout25*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+nao2, gout26*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+3)+nao2, gout27*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+4)+nao2, gout28*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+5)+nao2, gout29*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+2*nao2, gout30*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+2*nao2, gout31*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+2*nao2, gout32*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+3)+2*nao2, gout33*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+4)+2*nao2, gout34*d_0);
            atomicAdd(vk+(j0+0)+nao*(k0+5)+2*nao2, gout35*d_0);
            vk += 3*nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_ip1_jk_kernel_1000(GINTEnvVars envs, JKMatrix jk,
                                    BasisProdOffsets offsets)
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
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double * __restrict__ i_exponent = c_bpcache.a1;
    double * __restrict__ j_exponent = c_bpcache.a2;
    int ij, kl;
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
    double ABx = xi - bas_x[jsh];
    double ABy = yi - bas_y[jsh];
    double ABz = zi - bas_z[jsh];
    
    
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    double rw[4];
    int irys;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
        double alpha = i_exponent[ij];
        double beta = j_exponent[ij];
		    double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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

        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
            double weight0 = rw[irys+2] * fac;
            double root0 = rw[irys];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double B00 = u2 * tmp4;
            double tmp1 = 2 * B00;    
            double B10 = B00 + tmp4 * akl;
            double tmp2 = tmp1 * akl;
            double C00x = xij - xi - tmp2 * xijxkl;
            double C00y = yij - yi - tmp2 * yijykl;
            double C00z = zij - zi - tmp2 * zijzkl;
            
            

            double g0 = B10+C00x*C00x;
            double g1 = ABx+C00x;
            double g3 = B10+(C00x*g1);
            double g4 = B10+C00y*C00y;
            double g5 = ABy+C00y;
            double g7 = B10+(C00y*g5);
            double g8 = B10+C00z*C00z;
            double g9 = ABz+C00z;
            double g11 = B10+(C00z*g9);

            gout0 += (-1 + 2*alpha*g0) * weight0;
            gout1 += (2*alpha*C00x) * (C00y) * weight0;
            gout2 += (2*alpha*C00x) * (C00z) * weight0;
            gout3 += (C00x) * (2*alpha*C00y) * weight0;
            gout4 += (-1 + 2*alpha*g4) * weight0;
            gout5 += (2*alpha*C00y) * (C00z) * weight0;
            gout6 += (C00x) * (2*alpha*C00z) * weight0;
            gout7 += (C00y) * (2*alpha*C00z) * weight0;
            gout8 += (-1 + 2*alpha*g8) * weight0;
            gout9 += (2*beta*g3) * weight0;
            gout10 += (2*beta*g1) * (C00y) * weight0;
            gout11 += (2*beta*g1) * (C00z) * weight0;
            gout12 += (C00x) * (2*beta*g5) * weight0;
            gout13 += (2*beta*g7) * weight0;
            gout14 += (2*beta*g5) * (C00z) * weight0;
            gout15 += (C00x) * (2*beta*g9) * weight0;
            gout16 += (C00y) * (2*beta*g9) * weight0;
            gout17 += (2*beta*g11) * weight0;
        }
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double d_0, d_1, d_2;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+nao2, gout3*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+nao2, gout4*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+nao2, gout5*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+2*nao2, gout6*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+2*nao2, gout7*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+2*nao2, gout8*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0), gout9*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0), gout10*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0), gout11*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+nao2, gout12*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+nao2, gout13*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+nao2, gout14*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+2*nao2, gout15*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+2*nao2, gout16*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+2*nao2, gout17*d_0);
            vj += 3*nao2;
        }
        if (vk != NULL) {
            // ijkl, jk -> il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+nao2, gout3*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+nao2, gout4*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+nao2, gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+2*nao2, gout6*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+2*nao2, gout7*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+2*nao2, gout8*d_0);
            // ijkl, jl -> ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+nao2, gout3*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+nao2, gout4*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+nao2, gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+2*nao2, gout6*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+2*nao2, gout7*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+2*nao2, gout8*d_0);
            // ijkl, ik -> jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+2*nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            // ijkl, il -> jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+2*nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            vk += 3*nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_ip1_jk_kernel_1010(GINTEnvVars envs, JKMatrix jk,
                                    BasisProdOffsets offsets)
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
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double * __restrict__ i_exponent = c_bpcache.a1;
    double * __restrict__ j_exponent = c_bpcache.a2;
    int ij, kl;
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
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;

    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double ABx = xi - bas_x[jsh];
    double ABy = yi - bas_y[jsh];
    double ABz = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    double rw[4];
    int irys;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
        double alpha = i_exponent[ij];
        double beta = j_exponent[ij];
		    double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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

        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
            double weight0 = rw[irys+2] * fac;
            double root0 = rw[irys];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double B00 = u2 * tmp4;
            double tmp1 = 2 * B00;    
            double B10 = B00 + tmp4 * akl;
            double tmp2 = tmp1 * akl;
            double C00x = xij - xi - tmp2 * xijxkl;
            double C00y = yij - yi - tmp2 * yijykl;
            double C00z = zij - zi - tmp2 * zijzkl;
            
            double tmp3 = tmp1 * aij;
            double D00x = xkl - xk + tmp3 * xijxkl;
            double D00y = ykl - yk + tmp3 * yijykl;
            double D00z = zkl - zk + tmp3 * zijzkl;

            double g0 = B10+C00x*C00x;
            double g1 = C00x*D00x;
            double g2 = B00+g1;
            double g3 = 2*B00*C00x;
            double g4 = g0*D00x;
            double g5 = g3+g4;
            double g6 = ABx+C00x;
            double g8 = B10+(C00x*g6);
            double g10 = B00+(g6*D00x);
            double g15 = (B00*(ABx+(2*C00x)))+(g8*D00x);
            double g16 = B10+C00y*C00y;
            double g18 = B00+(C00y*D00y);
            double g21 = (2*B00*C00y)+(g16*D00y);
            double g22 = ABy+C00y;
            double g24 = B10+(C00y*g22);
            double g26 = B00+(g22*D00y);
            double g31 = (B00*(ABy+(2*C00y)))+(g24*D00y);
            double g32 = B10+C00z*C00z;
            double g34 = B00+(C00z*D00z);
            double g37 = (2*B00*C00z)+(g32*D00z);
            double g38 = ABz+C00z;
            double g40 = B10+(C00z*g38);
            double g42 = B00+(g38*D00z);
            double g47 = (B00*(ABz+(2*C00z)))+(g40*D00z);

            gout0 += (-D00x + 2*alpha*g5) * weight0;
            gout1 += (-1 + 2*alpha*g0) * (D00y) * weight0;
            gout2 += (-1 + 2*alpha*g0) * (D00z) * weight0;
            gout3 += (2*alpha*g2) * (C00y) * weight0;
            gout4 += (2*alpha*C00x) * (g18) * weight0;
            gout5 += (2*alpha*C00x) * (C00y) * (D00z) * weight0;
            gout6 += (2*alpha*g2) * (C00z) * weight0;
            gout7 += (2*alpha*C00x) * (D00y) * (C00z) * weight0;
            gout8 += (2*alpha*C00x) * (g34) * weight0;
            gout9 += (g2) * (2*alpha*C00y) * weight0;
            gout10 += (C00x) * (2*alpha*g18) * weight0;
            gout11 += (C00x) * (2*alpha*C00y) * (D00z) * weight0;
            gout12 += (D00x) * (-1 + 2*alpha*g16) * weight0;
            gout13 += (-D00y + 2*alpha*g21) * weight0;
            gout14 += (-1 + 2*alpha*g16) * (D00z) * weight0;
            gout15 += (D00x) * (2*alpha*C00y) * (C00z) * weight0;
            gout16 += (2*alpha*g18) * (C00z) * weight0;
            gout17 += (2*alpha*C00y) * (g34) * weight0;
            gout18 += (g2) * (2*alpha*C00z) * weight0;
            gout19 += (C00x) * (D00y) * (2*alpha*C00z) * weight0;
            gout20 += (C00x) * (2*alpha*g34) * weight0;
            gout21 += (D00x) * (C00y) * (2*alpha*C00z) * weight0;
            gout22 += (g18) * (2*alpha*C00z) * weight0;
            gout23 += (C00y) * (2*alpha*g34) * weight0;
            gout24 += (D00x) * (-1 + 2*alpha*g32) * weight0;
            gout25 += (D00y) * (-1 + 2*alpha*g32) * weight0;
            gout26 += (-D00z + 2*alpha*g37) * weight0;
            gout27 += (2*beta*g15) * weight0;
            gout28 += (2*beta*g8) * (D00y) * weight0;
            gout29 += (2*beta*g8) * (D00z) * weight0;
            gout30 += (2*beta*g10) * (C00y) * weight0;
            gout31 += (2*beta*g6) * (g18) * weight0;
            gout32 += (2*beta*g6) * (C00y) * (D00z) * weight0;
            gout33 += (2*beta*g10) * (C00z) * weight0;
            gout34 += (2*beta*g6) * (D00y) * (C00z) * weight0;
            gout35 += (2*beta*g6) * (g34) * weight0;
            gout36 += (g2) * (2*beta*g22) * weight0;
            gout37 += (C00x) * (2*beta*g26) * weight0;
            gout38 += (C00x) * (2*beta*g22) * (D00z) * weight0;
            gout39 += (D00x) * (2*beta*g24) * weight0;
            gout40 += (2*beta*g31) * weight0;
            gout41 += (2*beta*g24) * (D00z) * weight0;
            gout42 += (D00x) * (2*beta*g22) * (C00z) * weight0;
            gout43 += (2*beta*g26) * (C00z) * weight0;
            gout44 += (2*beta*g22) * (g34) * weight0;
            gout45 += (g2) * (2*beta*g38) * weight0;
            gout46 += (C00x) * (D00y) * (2*beta*g38) * weight0;
            gout47 += (C00x) * (2*beta*g42) * weight0;
            gout48 += (D00x) * (C00y) * (2*beta*g38) * weight0;
            gout49 += (g18) * (2*beta*g38) * weight0;
            gout50 += (C00y) * (2*beta*g42) * weight0;
            gout51 += (D00x) * (2*beta*g40) * weight0;
            gout52 += (D00y) * (2*beta*g40) * weight0;
            gout53 += (2*beta*g47) * weight0;
        }
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            d_0 = dm[(k0+0)+nao*(l0+0)];
            d_1 = dm[(k0+1)+nao*(l0+0)];
            d_2 = dm[(k0+2)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0+gout1*d_1+gout2*d_2);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout3*d_0+gout4*d_1+gout5*d_2);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout6*d_0+gout7*d_1+gout8*d_2);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+nao2, gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+2*nao2, gout18*d_0+gout19*d_1+gout20*d_2);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+2*nao2, gout21*d_0+gout22*d_1+gout23*d_2);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+2*nao2, gout24*d_0+gout25*d_1+gout26*d_2);
            atomicAdd(vj+nao*(i0+0)+(j0+0), gout27*d_0+gout28*d_1+gout29*d_2);
            atomicAdd(vj+nao*(i0+1)+(j0+0), gout30*d_0+gout31*d_1+gout32*d_2);
            atomicAdd(vj+nao*(i0+2)+(j0+0), gout33*d_0+gout34*d_1+gout35*d_2);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+nao2, gout36*d_0+gout37*d_1+gout38*d_2);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+nao2, gout39*d_0+gout40*d_1+gout41*d_2);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+nao2, gout42*d_0+gout43*d_1+gout44*d_2);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+2*nao2, gout45*d_0+gout46*d_1+gout47*d_2);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+2*nao2, gout48*d_0+gout49*d_1+gout50*d_2);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+2*nao2, gout51*d_0+gout52*d_1+gout53*d_2);
            vj += 3*nao2;
        }
        if (vk != NULL) {
            // ijkl, jk -> il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+0)+nao*(k0+1)];
            d_2 = dm[(j0+0)+nao*(k0+2)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0+gout1*d_1+gout2*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout3*d_0+gout4*d_1+gout5*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout6*d_0+gout7*d_1+gout8*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+nao2, gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+2*nao2, gout18*d_0+gout19*d_1+gout20*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+2*nao2, gout21*d_0+gout22*d_1+gout23*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+2*nao2, gout24*d_0+gout25*d_1+gout26*d_2);
            // ijkl, jl -> ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1), gout1*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2), gout2*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout3*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+1), gout4*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+2), gout5*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout6*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+1), gout7*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+2), gout8*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+nao2, gout9*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+nao2, gout10*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+nao2, gout11*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+nao2, gout12*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+1)+nao2, gout13*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+2)+nao2, gout14*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+nao2, gout15*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+1)+nao2, gout16*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+2)+nao2, gout17*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+2*nao2, gout18*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+1)+2*nao2, gout19*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+2)+2*nao2, gout20*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+2*nao2, gout21*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+1)+2*nao2, gout22*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+2)+2*nao2, gout23*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+2*nao2, gout24*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+1)+2*nao2, gout25*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+2)+2*nao2, gout26*d_0);
            // ijkl, ik -> jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+0)+nao*(k0+1)];
            d_2 = dm[(i0+0)+nao*(k0+2)];
            d_3 = dm[(i0+1)+nao*(k0+0)];
            d_4 = dm[(i0+1)+nao*(k0+1)];
            d_5 = dm[(i0+1)+nao*(k0+2)];
            d_6 = dm[(i0+2)+nao*(k0+0)];
            d_7 = dm[(i0+2)+nao*(k0+1)];
            d_8 = dm[(i0+2)+nao*(k0+2)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout27*d_0+gout28*d_1+gout29*d_2+gout30*d_3+gout31*d_4+gout32*d_5+gout33*d_6+gout34*d_7+gout35*d_8);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+nao2, gout36*d_0+gout37*d_1+gout38*d_2+gout39*d_3+gout40*d_4+gout41*d_5+gout42*d_6+gout43*d_7+gout44*d_8);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+2*nao2, gout45*d_0+gout46*d_1+gout47*d_2+gout48*d_3+gout49*d_4+gout50*d_5+gout51*d_6+gout52*d_7+gout53*d_8);
            // ijkl, il -> jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout27*d_0+gout30*d_1+gout33*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1), gout28*d_0+gout31*d_1+gout34*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2), gout29*d_0+gout32*d_1+gout35*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+nao2, gout36*d_0+gout39*d_1+gout42*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+nao2, gout37*d_0+gout40*d_1+gout43*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+nao2, gout38*d_0+gout41*d_1+gout44*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+2*nao2, gout45*d_0+gout48*d_1+gout51*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+1)+2*nao2, gout46*d_0+gout49*d_1+gout52*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+2)+2*nao2, gout47*d_0+gout50*d_1+gout53*d_2);
            vk += 3*nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_ip1_jk_kernel_1100(GINTEnvVars envs, JKMatrix jk,
                                    BasisProdOffsets offsets)
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
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double * __restrict__ i_exponent = c_bpcache.a1;
    double * __restrict__ j_exponent = c_bpcache.a2;
    int ij, kl;
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
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;

    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double ABx = xi - bas_x[jsh];
    double ABy = yi - bas_y[jsh];
    double ABz = zi - bas_z[jsh];
    
    
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    double rw[4];
    int irys;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
        double alpha = i_exponent[ij];
        double beta = j_exponent[ij];
		    double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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

        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
            double weight0 = rw[irys+2] * fac;
            double root0 = rw[irys];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double B00 = u2 * tmp4;
            double tmp1 = 2 * B00;    
            double B10 = B00 + tmp4 * akl;
            double tmp2 = tmp1 * akl;
            double C00x = xij - xi - tmp2 * xijxkl;
            double C00y = yij - yi - tmp2 * yijykl;
            double C00z = zij - zi - tmp2 * zijzkl;
            
            

            double g0 = B10+C00x*C00x;
            double g1 = ABx+C00x;
            double g2 = C00x*g1;
            double g3 = B10+g2;
            double g4 = 3*B10*C00x;
            double g6 = g4+C00x*C00x*C00x+(ABx*g0);
            double g7 = B10+g1*g1;
            double g10 = (ABx*ABx*C00x)+g4+C00x*C00x*C00x+(2*ABx*g0);
            double g11 = B10+C00y*C00y;
            double g12 = ABy+C00y;
            double g14 = B10+(C00y*g12);
            double g15 = 3*B10*C00y;
            double g17 = g15+C00y*C00y*C00y+(ABy*g11);
            double g18 = B10+g12*g12;
            double g21 = (ABy*ABy*C00y)+g15+C00y*C00y*C00y+(2*ABy*g11);
            double g22 = B10+C00z*C00z;
            double g23 = ABz+C00z;
            double g25 = B10+(C00z*g23);
            double g26 = 3*B10*C00z;
            double g28 = g26+C00z*C00z*C00z+(ABz*g22);
            double g29 = B10+g23*g23;
            double g32 = (ABz*ABz*C00z)+g26+C00z*C00z*C00z+(2*ABz*g22);

            gout0 += (-g1 + 2*alpha*g6) * weight0;
            gout1 += (-1 + 2*alpha*g0) * (g12) * weight0;
            gout2 += (-1 + 2*alpha*g0) * (g23) * weight0;
            gout3 += (2*alpha*g3) * (C00y) * weight0;
            gout4 += (2*alpha*C00x) * (g14) * weight0;
            gout5 += (2*alpha*C00x) * (C00y) * (g23) * weight0;
            gout6 += (2*alpha*g3) * (C00z) * weight0;
            gout7 += (2*alpha*C00x) * (g12) * (C00z) * weight0;
            gout8 += (2*alpha*C00x) * (g25) * weight0;
            gout9 += (g3) * (2*alpha*C00y) * weight0;
            gout10 += (C00x) * (2*alpha*g14) * weight0;
            gout11 += (C00x) * (2*alpha*C00y) * (g23) * weight0;
            gout12 += (g1) * (-1 + 2*alpha*g11) * weight0;
            gout13 += (-g12 + 2*alpha*g17) * weight0;
            gout14 += (-1 + 2*alpha*g11) * (g23) * weight0;
            gout15 += (g1) * (2*alpha*C00y) * (C00z) * weight0;
            gout16 += (2*alpha*g14) * (C00z) * weight0;
            gout17 += (2*alpha*C00y) * (g25) * weight0;
            gout18 += (g3) * (2*alpha*C00z) * weight0;
            gout19 += (C00x) * (g12) * (2*alpha*C00z) * weight0;
            gout20 += (C00x) * (2*alpha*g25) * weight0;
            gout21 += (g1) * (C00y) * (2*alpha*C00z) * weight0;
            gout22 += (g14) * (2*alpha*C00z) * weight0;
            gout23 += (C00y) * (2*alpha*g25) * weight0;
            gout24 += (g1) * (-1 + 2*alpha*g22) * weight0;
            gout25 += (g12) * (-1 + 2*alpha*g22) * weight0;
            gout26 += (-g23 + 2*alpha*g28) * weight0;
            gout27 += (-C00x + 2*beta*g10) * weight0;
            gout28 += (2*beta*g3) * (g12) * weight0;
            gout29 += (2*beta*g3) * (g23) * weight0;
            gout30 += (-1 + 2*beta*g7) * (C00y) * weight0;
            gout31 += (2*beta*g1) * (g14) * weight0;
            gout32 += (2*beta*g1) * (C00y) * (g23) * weight0;
            gout33 += (-1 + 2*beta*g7) * (C00z) * weight0;
            gout34 += (2*beta*g1) * (g12) * (C00z) * weight0;
            gout35 += (2*beta*g1) * (g25) * weight0;
            gout36 += (g3) * (2*beta*g12) * weight0;
            gout37 += (C00x) * (-1 + 2*beta*g18) * weight0;
            gout38 += (C00x) * (2*beta*g12) * (g23) * weight0;
            gout39 += (g1) * (2*beta*g14) * weight0;
            gout40 += (-C00y + 2*beta*g21) * weight0;
            gout41 += (2*beta*g14) * (g23) * weight0;
            gout42 += (g1) * (2*beta*g12) * (C00z) * weight0;
            gout43 += (-1 + 2*beta*g18) * (C00z) * weight0;
            gout44 += (2*beta*g12) * (g25) * weight0;
            gout45 += (g3) * (2*beta*g23) * weight0;
            gout46 += (C00x) * (g12) * (2*beta*g23) * weight0;
            gout47 += (C00x) * (-1 + 2*beta*g29) * weight0;
            gout48 += (g1) * (C00y) * (2*beta*g23) * weight0;
            gout49 += (g14) * (2*beta*g23) * weight0;
            gout50 += (C00y) * (-1 + 2*beta*g29) * weight0;
            gout51 += (g1) * (2*beta*g25) * weight0;
            gout52 += (g12) * (2*beta*g25) * weight0;
            gout53 += (-C00z + 2*beta*g32) * weight0;
        }
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double d_0, d_1, d_2;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+1), gout1*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+2), gout2*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout3*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+1), gout4*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+2), gout5*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout6*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+1), gout7*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+2), gout8*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+nao2, gout9*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+1)+nao2, gout10*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+2)+nao2, gout11*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+nao2, gout12*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+1)+nao2, gout13*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+2)+nao2, gout14*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+nao2, gout15*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+1)+nao2, gout16*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+2)+nao2, gout17*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+2*nao2, gout18*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+1)+2*nao2, gout19*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+2)+2*nao2, gout20*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+2*nao2, gout21*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+1)+2*nao2, gout22*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+2)+2*nao2, gout23*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+2*nao2, gout24*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+1)+2*nao2, gout25*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+2)+2*nao2, gout26*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0), gout27*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+1), gout28*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+2), gout29*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0), gout30*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+1), gout31*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+2), gout32*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0), gout33*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+1), gout34*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+2), gout35*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+nao2, gout36*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+1)+nao2, gout37*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+2)+nao2, gout38*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+nao2, gout39*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+1)+nao2, gout40*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+2)+nao2, gout41*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+nao2, gout42*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+1)+nao2, gout43*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+2)+nao2, gout44*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+2*nao2, gout45*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+1)+2*nao2, gout46*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+2)+2*nao2, gout47*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+2*nao2, gout48*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+1)+2*nao2, gout49*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+2)+2*nao2, gout50*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+2*nao2, gout51*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+1)+2*nao2, gout52*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+2)+2*nao2, gout53*d_0);
            vj += 3*nao2;
        }
        if (vk != NULL) {
            // ijkl, jk -> il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            d_1 = dm[(j0+1)+nao*(k0+0)];
            d_2 = dm[(j0+2)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0+gout1*d_1+gout2*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout3*d_0+gout4*d_1+gout5*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout6*d_0+gout7*d_1+gout8*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+nao2, gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+2*nao2, gout18*d_0+gout19*d_1+gout20*d_2);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+2*nao2, gout21*d_0+gout22*d_1+gout23*d_2);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+2*nao2, gout24*d_0+gout25*d_1+gout26*d_2);
            // ijkl, jl -> ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            d_1 = dm[(j0+1)+nao*(l0+0)];
            d_2 = dm[(j0+2)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0+gout1*d_1+gout2*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout3*d_0+gout4*d_1+gout5*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout6*d_0+gout7*d_1+gout8*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+nao2, gout9*d_0+gout10*d_1+gout11*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+nao2, gout12*d_0+gout13*d_1+gout14*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+nao2, gout15*d_0+gout16*d_1+gout17*d_2);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+2*nao2, gout18*d_0+gout19*d_1+gout20*d_2);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+2*nao2, gout21*d_0+gout22*d_1+gout23*d_2);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+2*nao2, gout24*d_0+gout25*d_1+gout26*d_2);
            // ijkl, ik -> jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout27*d_0+gout30*d_1+gout33*d_2);
            atomicAdd(vk+(j0+1)+nao*(l0+0), gout28*d_0+gout31*d_1+gout34*d_2);
            atomicAdd(vk+(j0+2)+nao*(l0+0), gout29*d_0+gout32*d_1+gout35*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+nao2, gout36*d_0+gout39*d_1+gout42*d_2);
            atomicAdd(vk+(j0+1)+nao*(l0+0)+nao2, gout37*d_0+gout40*d_1+gout43*d_2);
            atomicAdd(vk+(j0+2)+nao*(l0+0)+nao2, gout38*d_0+gout41*d_1+gout44*d_2);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+2*nao2, gout45*d_0+gout48*d_1+gout51*d_2);
            atomicAdd(vk+(j0+1)+nao*(l0+0)+2*nao2, gout46*d_0+gout49*d_1+gout52*d_2);
            atomicAdd(vk+(j0+2)+nao*(l0+0)+2*nao2, gout47*d_0+gout50*d_1+gout53*d_2);
            // ijkl, il -> jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout27*d_0+gout30*d_1+gout33*d_2);
            atomicAdd(vk+(j0+1)+nao*(k0+0), gout28*d_0+gout31*d_1+gout34*d_2);
            atomicAdd(vk+(j0+2)+nao*(k0+0), gout29*d_0+gout32*d_1+gout35*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+nao2, gout36*d_0+gout39*d_1+gout42*d_2);
            atomicAdd(vk+(j0+1)+nao*(k0+0)+nao2, gout37*d_0+gout40*d_1+gout43*d_2);
            atomicAdd(vk+(j0+2)+nao*(k0+0)+nao2, gout38*d_0+gout41*d_1+gout44*d_2);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+2*nao2, gout45*d_0+gout48*d_1+gout51*d_2);
            atomicAdd(vk+(j0+1)+nao*(k0+0)+2*nao2, gout46*d_0+gout49*d_1+gout52*d_2);
            atomicAdd(vk+(j0+2)+nao*(k0+0)+2*nao2, gout47*d_0+gout50*d_1+gout53*d_2);
            vk += 3*nao2;
        }
        dm += nao2;
    }
}

__global__
static void GINTint2e_ip1_jk_kernel_2000(GINTEnvVars envs, JKMatrix jk,
                                    BasisProdOffsets offsets)
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
    double norm = envs.fac;
    double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double * __restrict__ i_exponent = c_bpcache.a1;
    double * __restrict__ j_exponent = c_bpcache.a2;
    int ij, kl;
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
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;

    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double ABx = xi - bas_x[jsh];
    double ABy = yi - bas_y[jsh];
    double ABz = zi - bas_z[jsh];
    
    
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    double rw[4];
    int irys;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
        double alpha = i_exponent[ij];
        double beta = j_exponent[ij];
		    double aij = a12[ij];
        double eij = e12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
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

        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
            double weight0 = rw[irys+2] * fac;
            double root0 = rw[irys];
            double u2 = a0 * root0;
            double tmp4 = .5 / (u2 * aijkl + a1);
            double B00 = u2 * tmp4;
            double tmp1 = 2 * B00;    
            double B10 = B00 + tmp4 * akl;
            double tmp2 = tmp1 * akl;
            double C00x = xij - xi - tmp2 * xijxkl;
            double C00y = yij - yi - tmp2 * yijykl;
            double C00z = zij - zi - tmp2 * zijzkl;
            
            

            double g0 = B10+C00x*C00x;
            double g1 = 3*B10*C00x;
            double g2 = g1+C00x*C00x*C00x;
            double g3 = ABx+C00x;
            double g5 = B10+(C00x*g3);
            double g7 = g1+C00x*C00x*C00x+(ABx*g0);
            double g8 = B10+C00y*C00y;
            double g9 = 3*B10*C00y;
            double g10 = g9+C00y*C00y*C00y;
            double g11 = ABy+C00y;
            double g13 = B10+(C00y*g11);
            double g15 = g9+C00y*C00y*C00y+(ABy*g8);
            double g16 = B10+C00z*C00z;
            double g17 = 3*B10*C00z;
            double g18 = g17+C00z*C00z*C00z;
            double g19 = ABz+C00z;
            double g21 = B10+(C00z*g19);
            double g23 = g17+C00z*C00z*C00z+(ABz*g16);

            gout0 += (-2*C00x + 2*alpha*g2) * weight0;
            gout1 += (-1 + 2*alpha*g0) * (C00y) * weight0;
            gout2 += (-1 + 2*alpha*g0) * (C00z) * weight0;
            gout3 += (2*alpha*C00x) * (g8) * weight0;
            gout4 += (2*alpha*C00x) * (C00y) * (C00z) * weight0;
            gout5 += (2*alpha*C00x) * (g16) * weight0;
            gout6 += (g0) * (2*alpha*C00y) * weight0;
            gout7 += (C00x) * (-1 + 2*alpha*g8) * weight0;
            gout8 += (C00x) * (2*alpha*C00y) * (C00z) * weight0;
            gout9 += (-2*C00y + 2*alpha*g10) * weight0;
            gout10 += (-1 + 2*alpha*g8) * (C00z) * weight0;
            gout11 += (2*alpha*C00y) * (g16) * weight0;
            gout12 += (g0) * (2*alpha*C00z) * weight0;
            gout13 += (C00x) * (C00y) * (2*alpha*C00z) * weight0;
            gout14 += (C00x) * (-1 + 2*alpha*g16) * weight0;
            gout15 += (g8) * (2*alpha*C00z) * weight0;
            gout16 += (C00y) * (-1 + 2*alpha*g16) * weight0;
            gout17 += (-2*C00z + 2*alpha*g18) * weight0;
            gout18 += (2*beta*g7) * weight0;
            gout19 += (2*beta*g5) * (C00y) * weight0;
            gout20 += (2*beta*g5) * (C00z) * weight0;
            gout21 += (2*beta*g3) * (g8) * weight0;
            gout22 += (2*beta*g3) * (C00y) * (C00z) * weight0;
            gout23 += (2*beta*g3) * (g16) * weight0;
            gout24 += (g0) * (2*beta*g11) * weight0;
            gout25 += (C00x) * (2*beta*g13) * weight0;
            gout26 += (C00x) * (2*beta*g11) * (C00z) * weight0;
            gout27 += (2*beta*g15) * weight0;
            gout28 += (2*beta*g13) * (C00z) * weight0;
            gout29 += (2*beta*g11) * (g16) * weight0;
            gout30 += (g0) * (2*beta*g19) * weight0;
            gout31 += (C00x) * (C00y) * (2*beta*g19) * weight0;
            gout32 += (C00x) * (2*beta*g21) * weight0;
            gout33 += (g8) * (2*beta*g19) * weight0;
            gout34 += (C00y) * (2*beta*g21) * weight0;
            gout35 += (2*beta*g23) * weight0;
        }
    } }

    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double* __restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double d_0, d_1, d_2, d_3, d_4, d_5;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            d_0 = dm[(k0+0)+nao*(l0+0)];
            atomicAdd(vj+(i0+0)+nao*(j0+0), gout0*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0), gout1*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0), gout2*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+0), gout3*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+0), gout4*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+0), gout5*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+nao2, gout6*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+nao2, gout7*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+nao2, gout8*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+0)+nao2, gout9*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+0)+nao2, gout10*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+0)+nao2, gout11*d_0);
            atomicAdd(vj+(i0+0)+nao*(j0+0)+2*nao2, gout12*d_0);
            atomicAdd(vj+(i0+1)+nao*(j0+0)+2*nao2, gout13*d_0);
            atomicAdd(vj+(i0+2)+nao*(j0+0)+2*nao2, gout14*d_0);
            atomicAdd(vj+(i0+3)+nao*(j0+0)+2*nao2, gout15*d_0);
            atomicAdd(vj+(i0+4)+nao*(j0+0)+2*nao2, gout16*d_0);
            atomicAdd(vj+(i0+5)+nao*(j0+0)+2*nao2, gout17*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0), gout18*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0), gout19*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0), gout20*d_0);
            atomicAdd(vj+nao*(i0+3)+(j0+0), gout21*d_0);
            atomicAdd(vj+nao*(i0+4)+(j0+0), gout22*d_0);
            atomicAdd(vj+nao*(i0+5)+(j0+0), gout23*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+nao2, gout24*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+nao2, gout25*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+nao2, gout26*d_0);
            atomicAdd(vj+nao*(i0+3)+(j0+0)+nao2, gout27*d_0);
            atomicAdd(vj+nao*(i0+4)+(j0+0)+nao2, gout28*d_0);
            atomicAdd(vj+nao*(i0+5)+(j0+0)+nao2, gout29*d_0);
            atomicAdd(vj+nao*(i0+0)+(j0+0)+2*nao2, gout30*d_0);
            atomicAdd(vj+nao*(i0+1)+(j0+0)+2*nao2, gout31*d_0);
            atomicAdd(vj+nao*(i0+2)+(j0+0)+2*nao2, gout32*d_0);
            atomicAdd(vj+nao*(i0+3)+(j0+0)+2*nao2, gout33*d_0);
            atomicAdd(vj+nao*(i0+4)+(j0+0)+2*nao2, gout34*d_0);
            atomicAdd(vj+nao*(i0+5)+(j0+0)+2*nao2, gout35*d_0);
            vj += 3*nao2;
        }
        if (vk != NULL) {
            // ijkl, jk -> il
            d_0 = dm[(j0+0)+nao*(k0+0)];
            atomicAdd(vk+(i0+0)+nao*(l0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0), gout2*d_0);
            atomicAdd(vk+(i0+3)+nao*(l0+0), gout3*d_0);
            atomicAdd(vk+(i0+4)+nao*(l0+0), gout4*d_0);
            atomicAdd(vk+(i0+5)+nao*(l0+0), gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+nao2, gout6*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+nao2, gout7*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+nao2, gout8*d_0);
            atomicAdd(vk+(i0+3)+nao*(l0+0)+nao2, gout9*d_0);
            atomicAdd(vk+(i0+4)+nao*(l0+0)+nao2, gout10*d_0);
            atomicAdd(vk+(i0+5)+nao*(l0+0)+nao2, gout11*d_0);
            atomicAdd(vk+(i0+0)+nao*(l0+0)+2*nao2, gout12*d_0);
            atomicAdd(vk+(i0+1)+nao*(l0+0)+2*nao2, gout13*d_0);
            atomicAdd(vk+(i0+2)+nao*(l0+0)+2*nao2, gout14*d_0);
            atomicAdd(vk+(i0+3)+nao*(l0+0)+2*nao2, gout15*d_0);
            atomicAdd(vk+(i0+4)+nao*(l0+0)+2*nao2, gout16*d_0);
            atomicAdd(vk+(i0+5)+nao*(l0+0)+2*nao2, gout17*d_0);
            // ijkl, jl -> ik
            d_0 = dm[(j0+0)+nao*(l0+0)];
            atomicAdd(vk+(i0+0)+nao*(k0+0), gout0*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0), gout1*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0), gout2*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+0), gout3*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+0), gout4*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+0), gout5*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+nao2, gout6*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+nao2, gout7*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+nao2, gout8*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+0)+nao2, gout9*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+0)+nao2, gout10*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+0)+nao2, gout11*d_0);
            atomicAdd(vk+(i0+0)+nao*(k0+0)+2*nao2, gout12*d_0);
            atomicAdd(vk+(i0+1)+nao*(k0+0)+2*nao2, gout13*d_0);
            atomicAdd(vk+(i0+2)+nao*(k0+0)+2*nao2, gout14*d_0);
            atomicAdd(vk+(i0+3)+nao*(k0+0)+2*nao2, gout15*d_0);
            atomicAdd(vk+(i0+4)+nao*(k0+0)+2*nao2, gout16*d_0);
            atomicAdd(vk+(i0+5)+nao*(k0+0)+2*nao2, gout17*d_0);
            // ijkl, ik -> jl
            d_0 = dm[(i0+0)+nao*(k0+0)];
            d_1 = dm[(i0+1)+nao*(k0+0)];
            d_2 = dm[(i0+2)+nao*(k0+0)];
            d_3 = dm[(i0+3)+nao*(k0+0)];
            d_4 = dm[(i0+4)+nao*(k0+0)];
            d_5 = dm[(i0+5)+nao*(k0+0)];
            atomicAdd(vk+(j0+0)+nao*(l0+0), gout18*d_0+gout19*d_1+gout20*d_2+gout21*d_3+gout22*d_4+gout23*d_5);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+nao2, gout24*d_0+gout25*d_1+gout26*d_2+gout27*d_3+gout28*d_4+gout29*d_5);
            atomicAdd(vk+(j0+0)+nao*(l0+0)+2*nao2, gout30*d_0+gout31*d_1+gout32*d_2+gout33*d_3+gout34*d_4+gout35*d_5);
            // ijkl, il -> jk
            d_0 = dm[(i0+0)+nao*(l0+0)];
            d_1 = dm[(i0+1)+nao*(l0+0)];
            d_2 = dm[(i0+2)+nao*(l0+0)];
            d_3 = dm[(i0+3)+nao*(l0+0)];
            d_4 = dm[(i0+4)+nao*(l0+0)];
            d_5 = dm[(i0+5)+nao*(l0+0)];
            atomicAdd(vk+(j0+0)+nao*(k0+0), gout18*d_0+gout19*d_1+gout20*d_2+gout21*d_3+gout22*d_4+gout23*d_5);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+nao2, gout24*d_0+gout25*d_1+gout26*d_2+gout27*d_3+gout28*d_4+gout29*d_5);
            atomicAdd(vk+(j0+0)+nao*(k0+0)+2*nao2, gout30*d_0+gout31*d_1+gout32*d_2+gout33*d_3+gout34*d_4+gout35*d_5);
            vk += 3*nao2;
        }
        dm += nao2;
    }
}
