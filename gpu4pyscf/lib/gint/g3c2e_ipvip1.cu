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

template <int LI, int LJ, int LK, int NROOTS> __device__
static void GINTgout3c2e_ipvip1(GINTEnvVars envs, double* __restrict__ gout, double *g0, double ai2, double aj2)
{
    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;

    for (int ik = 0, i = 0; ik < nfk; ik++){
    for (int ij = 0; ij < nfj; ij++){
    for (int ii = 0; ii < nfi; ii++, i++){
        const int loc_k = c_l_locs[LK] + ik;
        const int loc_j = c_l_locs[LJ] + ij;
        const int loc_i = c_l_locs[LI] + ii;

        int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
        int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
        int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

        const int i_idx = c_idx[loc_i];
        const int i_idy = c_idy[loc_i];
        const int i_idz = c_idz[loc_i];

        const int j_idx = c_idx[loc_j];
        const int j_idy = c_idy[loc_j];
        const int j_idz = c_idz[loc_j];

        double sxx = gout[9*i + 0];
        double sxy = gout[9*i + 1];
        double sxz = gout[9*i + 2];
        double syx = gout[9*i + 3];
        double syy = gout[9*i + 4];
        double syz = gout[9*i + 5];
        double szx = gout[9*i + 6];
        double szy = gout[9*i + 7];
        double szz = gout[9*i + 8];
#pragma unroll
        for (int n = 0; n < NROOTS; ++n, ++ix, ++iy, ++iz) {
            const double g0_x = g0[ix];
            const double g0_y = g0[iy];
            const double g0_z = g0[iz];

            // g1
            double g1_x = aj2*g0[ix+dj];
            double g1_y = aj2*g0[iy+dj];
            double g1_z = aj2*g0[iz+dj];
            g1_x += j_idx>0 ? j_idx*g0[ix-dj] : 0.0;
            g1_y += j_idy>0 ? j_idy*g0[iy-dj] : 0.0;
            g1_z += j_idz>0 ? j_idz*g0[iz-dj] : 0.0;

            // g2
            double g2_x = ai2*g0[ix+di];
            double g2_y = ai2*g0[iy+di];
            double g2_z = ai2*g0[iz+di];
            g2_x += i_idx>0 ? i_idx*g0[ix-di] : 0.0;
            g2_y += i_idy>0 ? i_idy*g0[iy-di] : 0.0;
            g2_z += i_idz>0 ? i_idz*g0[iz-di] : 0.0;

            // g3 
            double g3_x = ai2*g0[ix+di+dj];
            double g3_y = ai2*g0[iy+di+dj];
            double g3_z = ai2*g0[iz+di+dj];
            if (i_idx > 0) { g3_x += i_idx*g0[ix-di+dj]; }
            if (i_idy > 0) { g3_y += i_idy*g0[iy-di+dj]; }
            if (i_idz > 0) { g3_z += i_idz*g0[iz-di+dj]; }
            g3_x *= aj2;
            g3_y *= aj2;
            g3_z *= aj2;
            if (j_idx > 0)              { g3_x += ai2 * j_idx * g0[ix+di-dj]; }
            if (j_idy > 0)              { g3_y += ai2 * j_idy * g0[iy+di-dj]; }
            if (j_idz > 0)              { g3_z += ai2 * j_idz * g0[iz+di-dj]; }
            if (i_idx > 0 && j_idx > 0) { g3_x += i_idx * j_idx * g0[ix-di-dj]; }
            if (i_idy > 0 && j_idy > 0) { g3_y += i_idy * j_idy * g0[iy-di-dj]; }
            if (i_idz > 0 && j_idz > 0) { g3_z += i_idz * j_idz * g0[iz-di-dj]; }

            sxx += g3_x * g0_y * g0_z;
            sxy += g2_x * g1_y * g0_z;
            sxz += g2_x * g0_y * g1_z;
            syx += g1_x * g2_y * g0_z;
            syy += g0_x * g3_y * g0_z;
            syz += g0_x * g2_y * g1_z;
            szx += g1_x * g0_y * g2_z;
            szy += g0_x * g1_y * g2_z;
            szz += g0_x * g0_y * g3_z;
        }

        gout[9*i + 0] = sxx;
        gout[9*i + 1] = sxy;
        gout[9*i + 2] = sxz;
        gout[9*i + 3] = syx;
        gout[9*i + 4] = syy;
        gout[9*i + 5] = syz;
        gout[9*i + 6] = szx;
        gout[9*i + 7] = szy;
        gout[9*i + 8] = szz;
    }}}
}

// Unrolled version
template <int LI, int LJ, int LK> __global__
void GINTfill_int3c2e_ipvip1_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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

    constexpr int LI_CEIL = LI + 1;
    constexpr int LJ_CEIL = LJ + 1;
    constexpr int NROOTS = (LI_CEIL+LJ_CEIL+LK)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI_CEIL+1)*(LJ_CEIL+1)*(LK+1);
    
    double g0[GSIZE];
    
    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;
    double gout[9*nfi*nfj*nfk] = {0};
    
    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e<LI_CEIL, LJ_CEIL, LK>(envs, g0, norm, as_ish, as_jsh, ksh, ij, kl);
        const double ai2 = -2.0*c_bpcache.a1[ij];
        const double aj2 = -2.0*c_bpcache.a2[ij];
        GINTgout3c2e_ipvip1<LI,LJ,LK,NROOTS>(envs, gout, g0, ai2, aj2);
    } }
    GINTwrite_int3c2e_ipip(eri, gout, as_ish, as_jsh, ksh);
}

template <int NROOTS> __device__
static void GINTwrite_int3c2e_ipvip1_direct(GINTEnvVars envs, ERITensor eri, 
    double* __restrict__ g0, double ai2, double aj2, 
    const int ish, const int jsh, const int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;
    const int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    const int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    const int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    const int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;

    int * __restrict__ c_idy = c_idx + TOT_NF;
    int * __restrict__ c_idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;

    for (int k = 0; k < k1-k0; ++k) {
    for (int j = 0; j < j1-j0; ++j) {
    for (int i = 0; i < i1-i0; ++i) {
        const int loc_k = c_l_locs[lk] + k;
        const int loc_j = c_l_locs[lj] + j;
        const int loc_i = c_l_locs[li] + i;

        int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
        int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
        int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;
        
        const int i_idx = c_idx[loc_i];
        const int i_idy = c_idy[loc_i];
        const int i_idz = c_idz[loc_i];

        const int j_idx = c_idx[loc_j];
        const int j_idy = c_idy[loc_j];
        const int j_idz = c_idz[loc_j];

        double eri_xx = 0;
        double eri_xy = 0;
        double eri_xz = 0;
        double eri_yx = 0;
        double eri_yy = 0;
        double eri_yz = 0;
        double eri_zx = 0;
        double eri_zy = 0;
        double eri_zz = 0;
        for (int ir = 0; ir < NROOTS; ++ir, ++ix, ++iy, ++iz){
            double g0_x = g0[ix];
            double g0_y = g0[iy];
            double g0_z = g0[iz];

            // g1
            double g1_x = aj2*g0[ix+dj];
            double g1_y = aj2*g0[iy+dj];
            double g1_z = aj2*g0[iz+dj];
            g1_x += j_idx>0 ? j_idx*g0[ix-dj] : 0.0;
            g1_y += j_idy>0 ? j_idy*g0[iy-dj] : 0.0;
            g1_z += j_idz>0 ? j_idz*g0[iz-dj] : 0.0;

            // g2
            double g2_x = ai2*g0[ix+di];
            double g2_y = ai2*g0[iy+di];
            double g2_z = ai2*g0[iz+di];
            g2_x += i_idx>0 ? i_idx*g0[ix-di] : 0.0;
            g2_y += i_idy>0 ? i_idy*g0[iy-di] : 0.0;
            g2_z += i_idz>0 ? i_idz*g0[iz-di] : 0.0;

            // g3 
            double g3_x = ai2*g0[ix+di+dj];
            double g3_y = ai2*g0[iy+di+dj];
            double g3_z = ai2*g0[iz+di+dj];
            if (i_idx > 0) { g3_x += i_idx*g0[ix-di+dj]; }
            if (i_idy > 0) { g3_y += i_idy*g0[iy-di+dj]; }
            if (i_idz > 0) { g3_z += i_idz*g0[iz-di+dj]; }
            g3_x *= aj2;
            g3_y *= aj2;
            g3_z *= aj2;
            if (j_idx > 0)              { g3_x += ai2 * j_idx * g0[ix+di-dj]; }
            if (j_idy > 0)              { g3_y += ai2 * j_idy * g0[iy+di-dj]; }
            if (j_idz > 0)              { g3_z += ai2 * j_idz * g0[iz+di-dj]; }
            if (i_idx > 0 && j_idx > 0) { g3_x += i_idx * j_idx * g0[ix-di-dj]; }
            if (i_idy > 0 && j_idy > 0) { g3_y += i_idy * j_idy * g0[iy-di-dj]; }
            if (i_idz > 0 && j_idz > 0) { g3_z += i_idz * j_idz * g0[iz-di-dj]; }

            eri_xx += g3_x * g0_y * g0_z;
            eri_xy += g2_x * g1_y * g0_z;
            eri_xz += g2_x * g0_y * g1_z;
            eri_yx += g1_x * g2_y * g0_z;
            eri_yy += g0_x * g3_y * g0_z;
            eri_yz += g0_x * g2_y * g1_z;
            eri_zx += g1_x * g0_y * g2_z;
            eri_zy += g0_x * g1_y * g2_z;
            eri_zz += g0_x * g0_y * g3_z;
        }
        int off = (i+i0) + jstride*(j+j0) + (k+k0)*kstride;
        double *eri_data = eri.data + off;
        eri_data[0 * lstride] += eri_xx;
        eri_data[1 * lstride] += eri_xy;
        eri_data[2 * lstride] += eri_xz;
        eri_data[3 * lstride] += eri_yx;
        eri_data[4 * lstride] += eri_yy;
        eri_data[5 * lstride] += eri_yz;
        eri_data[6 * lstride] += eri_zx;
        eri_data[7 * lstride] += eri_zy;
        eri_data[8 * lstride] += eri_zz;
    }}}
}

// General version
template <int NROOTS, int GSIZE> __global__
void GINTfill_int3c2e_ipvip1_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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

    double g0[GSIZE];
    
    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e<NROOTS>(envs, g0, norm, as_ish, as_jsh, ksh, ij, kl);
        const double ai2 = -2.0*c_bpcache.a1[ij];
        const double aj2 = -2.0*c_bpcache.a2[ij];
        GINTwrite_int3c2e_ipvip1_direct<NROOTS>(envs, eri, g0, ai2, aj2, ish, jsh, ksh);
    } }
}


__global__
static void GINTfill_int3c2e_ipvip1_kernel000(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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
    double* __restrict__ exp_ket = c_bpcache.a2;
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
    const double xixj = xi - bas_x[jsh];
    const double yiyj = yi - bas_y[jsh];
    const double zizj = zi - bas_z[jsh];
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
        const double aj2 = -2.0*exp_ket[ij];
        
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
            const double b10 = b00 + tmp4 * akl;
            const double c00x = xij - xi - tmp2 * xijxkl;
            const double c00y = yij - yi - tmp2 * yijykl;
            const double c00z = zij - zi - tmp2 * zijzkl;
            const double g_0 = 1;
            const double g_1 = c00x;
            const double g_2 = c00x + xixj;
            const double g_3 = c00x * (c00x + xixj) + b10;
            const double g_4 = 1;
            const double g_5 = c00y;
            const double g_6 = c00y + yiyj;
            const double g_7 = c00y * (c00y + yiyj) + b10;
            const double g_8 = weight0 * fac;
            const double g_9 = c00z * g_8;
            const double g_10 = g_8 * (c00z + zizj);
            const double g_11 = b10 * g_8 + c00z * g_9 + zizj * g_9;
            
            const double dg_3 = ai2 * aj2 * g_3;
            const double dg_2 = aj2 * g_1;
            const double dg_1 = ai2 * g_2;

            const double dg_7 = ai2 * aj2 * g_7;
            const double dg_6 = aj2 * g_5;
            const double dg_5 = ai2 * g_6;

            const double dg_11 = ai2 * aj2 * g_11;
            const double dg_10 = aj2 * g_9;
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
    const int *ao_loc = c_bpcache.ao_loc;
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
