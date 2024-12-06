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

// Unrolled version
template <int LI, int LJ, int LK> __global__
void GINTfill_int3c2e_ip1ip2_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    const double norm = envs.fac;
    const int bas_ij = offsets.bas_ij + task_ij;
    const int bas_kl = offsets.bas_kl + task_kl;
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];

    double* __restrict__ exp_bra = c_bpcache.a1;
    constexpr int LI_CEIL = LI + 1;
    constexpr int LK_CEIL = LK + 1;
    constexpr int NROOTS = (LI_CEIL+LJ+LK_CEIL)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI_CEIL+1)*(LJ+1)*(LK_CEIL+1);
    
    double g0[4*GSIZE];
    double * __restrict__ g1 = g0 + GSIZE;
    double * __restrict__ g2 = g1 + GSIZE;
    double * __restrict__ g3 = g2 + GSIZE;

    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;
    double gout[9*nfi*nfj*nfk] = {0};

    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_int3c2e<LI_CEIL, LJ, LK_CEIL>(envs, g0, norm, as_ish, as_jsh, ksh, ij, kl);

            const double ai2 = -2.0*exp_bra[ij];
            const double ak2 = -2.0*exp_bra[kl];
            GINTnabla1k_2e<LI+1,LJ,LK,NROOTS>(envs, g1, g0, ak2);
            GINTnabla1i_2e<LI,  LJ,LK,NROOTS>(envs, g2, g0, ai2);
            GINTnabla1i_2e<LI,  LJ,LK,NROOTS>(envs, g3, g1, ai2);
            //GINTwrite_int3c2e_ipip_direct<LI, LJ, LK>(envs, eri, g0, g1, g2, g3, ish, jsh, ksh);
            GINTgout3c2e_ipip<LI,LJ,LK,NROOTS>(envs, gout, g0, g1, g2, g3);
    } }
    GINTwrite_int3c2e_ipip(eri, gout, as_ish, as_jsh, ksh);
}

// General version
template <int NROOTS, int GSIZE> __global__
void GINTfill_int3c2e_ip1ip2_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    const double norm = envs.fac;
    const int bas_ij = offsets.bas_ij + task_ij;
    const int bas_kl = offsets.bas_kl + task_kl;
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];

    double g0[4*GSIZE];
    double * __restrict__ g1 = g0 + GSIZE;
    double * __restrict__ g2 = g1 + GSIZE;
    double * __restrict__ g3 = g2 + GSIZE;
    double* __restrict__ exp_bra = c_bpcache.a1;
    
    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_int3c2e<NROOTS>(envs, g0, norm, as_ish, as_jsh, ksh, ij, kl);

            double ai2 = -2.0*exp_bra[ij];
            double ak2 = -2.0*exp_bra[kl];
            GINTnabla1k_2e<NROOTS>(envs, g1, g0, ak2, envs.i_l+1, envs.j_l, envs.k_l);
            GINTnabla1i_2e<NROOTS>(envs, g2, g0, ai2, envs.i_l,   envs.j_l, envs.k_l);
            GINTnabla1i_2e<NROOTS>(envs, g3, g1, ai2, envs.i_l,   envs.j_l, envs.k_l);
            GINTwrite_int3c2e_ipip_direct<NROOTS>(envs, eri, g0, g1, g2, g3, ish, jsh, ksh);
    } }
}


__global__
static void GINTfill_int3c2e_ip1ip2_kernel000(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int bas_kl = offsets.bas_kl + task_kl;
    const double norm = envs.fac;
    const double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];
    double* __restrict__ exp_bra = c_bpcache.a1;
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;

    const int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gxx = 0;
    double gxy = 0;
    double gxz = 0;
    double gyx = 0;
    double gyy = 0;
    double gyz = 0;
    double gzx = 0;
    double gzy = 0;
    double gzz = 0;
    const double xi = bas_x[ish];
    const double yi = bas_y[ish];
    const double zi = bas_z[ish];
    const double xk = bas_x[ksh];
    const double yk = bas_y[ksh];
    const double zk = bas_z[ksh];
    const int prim_ij0 = prim_ij;
    const int prim_ij1 = prim_ij + nprim_ij;
    const int prim_kl0 = prim_kl;
    const int prim_kl1 = prim_kl + nprim_kl;
    for (int ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (int kl = prim_kl0; kl < prim_kl1; ++kl) {
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double xij = x12[ij];
        const double yij = y12[ij];
        const double zij = z12[ij];
        const double akl = a12[kl];
        const double ekl = e12[kl];
        const double xkl = x12[kl];
        const double ykl = y12[kl];
        const double zkl = z12[kl];
        const double xijxkl = xij - xkl;
        const double yijykl = yij - ykl;
        const double zijzkl = zij - zkl;
        const double aijkl = aij + akl;
        const double a1 = aij * akl;
        double a0 = a1 / aijkl;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
        a0 *= theta;
        const double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
        const double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
        //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

        const double ai2 = -2.0*exp_bra[ij];
        const double ak2 = -2.0*exp_bra[kl];

        double rw[4];
        GINTrys_root<2>(x, rw);
        GINTscale_u<2>(rw, theta);
        for (int irys = 0; irys < 2; ++irys) {
            const double root0 = rw[irys];
            const double weight0 = rw[irys+2];
            const double u2 = a0 * root0;
            const double tmp4 = .5 / (u2 * aijkl + a1);
            const double b00 = u2 * tmp4;
            const double tmp1 = 2 * b00;
            const double tmp2 = tmp1 * akl;
            const double c00x = xij - xi - tmp2 * xijxkl;
            const double c00y = yij - yi - tmp2 * yijykl;
            const double c00z = zij - zi - tmp2 * zijzkl;
            const double tmp3 = tmp1 * aij;
            const double c0px = xkl - xk + tmp3 * xijxkl;
            const double c0py = ykl - yk + tmp3 * yijykl;
            const double c0pz = zkl - zk + tmp3 * zijzkl;
            const double g_0 = 1;
            const double g_1 = c00x;
            const double g_2 = c0px;
            const double g_3 = c0px * c00x + b00;
            const double g_4 = 1;
            const double g_5 = c00y;
            const double g_6 = c0py;
            const double g_7 = c0py * c00y + b00;
            const double g_8 = weight0 * fac;
            const double g_9 = c00z * g_8;
            const double g_10 = c0pz * g_8;
            const double g_11 = b00 * g_8 + c0pz * g_9;

            const double dg_3 = ai2 * ak2 * g_3;
            const double dg_2 = ak2 * g_1;
            const double dg_1 = ai2 * g_2;

            const double dg_7 = ai2 * ak2 * g_7;
            const double dg_6 = ak2 * g_5;
            const double dg_5 = ai2 * g_6;

            const double dg_11 = ai2 * ak2 * g_11;
            const double dg_10 = ak2 * g_9;
            const double dg_9 = ai2 * g_10;

            gxx += dg_3 * g_4  * g_8;
            gxy += dg_2 * dg_5 * g_8;
            gxz += dg_2 * g_4  * dg_9;
            gyx += dg_1 * dg_6 * g_8;
            gyy += g_0  * dg_7 * g_8;
            gyz += g_0  * dg_6 * dg_9;
            gzx += dg_1 * g_4  * dg_10;
            gzy += g_0  * dg_5 * dg_10;
            gzz += g_0  * g_4  * dg_11;
        }
    } }

    const int jstride = eri.stride_j;
    const int kstride = eri.stride_k;
    const int lstride = eri.stride_l;
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh] - eri.ao_offsets_k;

    double* __restrict__ eri_ij = eri.data + k0*kstride+j0*jstride+i0;
 
    eri_ij[0*lstride] = gxx;
    eri_ij[1*lstride] = gxy;
    eri_ij[2*lstride] = gxz;
    eri_ij[3*lstride] = gyx;
    eri_ij[4*lstride] = gyy;
    eri_ij[5*lstride] = gyz;
    eri_ij[6*lstride] = gzx;
    eri_ij[7*lstride] = gzy;
    eri_ij[8*lstride] = gzz;
}