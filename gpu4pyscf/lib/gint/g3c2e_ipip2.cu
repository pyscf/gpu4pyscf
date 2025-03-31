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
static void GINTgout3c2e_ipip2(GINTEnvVars envs, double* __restrict__ gout, double *g0, double ak2)
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

        const int k_idx = c_idx[loc_k];
        const int k_idy = c_idy[loc_k];
        const int k_idz = c_idz[loc_k];

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

            // g2
            double g2_x = ak2*g0[ix+dk];
            double g2_y = ak2*g0[iy+dk];
            double g2_z = ak2*g0[iz+dk];
            g2_x += k_idx>0 ? k_idx*g0[ix-dk] : 0.0;
            g2_y += k_idy>0 ? k_idy*g0[iy-dk] : 0.0;
            g2_z += k_idz>0 ? k_idz*g0[iz-dk] : 0.0;

            // g3
            double g3_x = ak2*g0[ix+2*dk];
            double g3_y = ak2*g0[iy+2*dk];
            double g3_z = ak2*g0[iz+2*dk];
            g3_x += (k_idx+1)*g0[ix];
            g3_y += (k_idy+1)*g0[iy];
            g3_z += (k_idz+1)*g0[iz];
            g3_x *= ak2;
            g3_y *= ak2;
            g3_z *= ak2;
            if (k_idx > 0) { g3_x += ak2 * k_idx * g0[ix]; }
            if (k_idy > 0) { g3_y += ak2 * k_idy * g0[iy]; }
            if (k_idz > 0) { g3_z += ak2 * k_idz * g0[iz]; }
            if (k_idx > 1) { g3_x += k_idx * (k_idx-1) * g0[ix-2*dk]; }
            if (k_idy > 1) { g3_y += k_idy * (k_idy-1) * g0[iy-2*dk]; }
            if (k_idz > 1) { g3_z += k_idz * (k_idz-1) * g0[iz-2*dk]; }

            sxx += g3_x * g0_y * g0_z;
            sxy += g2_x * g2_y * g0_z;
            sxz += g2_x * g0_y * g2_z;
            syx += g2_x * g2_y * g0_z;
            syy += g0_x * g3_y * g0_z;
            syz += g0_x * g2_y * g2_z;
            szx += g2_x * g0_y * g2_z;
            szy += g0_x * g2_y * g2_z;
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
void GINTfill_int3c2e_ipip2_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    const int task_ij = blockIdx.x;// * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y;// * blockDim.y + threadIdx.y;

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
    const int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];

    constexpr int LK_CEIL = LK + 2;
    constexpr int NROOTS = (LI+LJ+LK_CEIL)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI+1)*(LJ+1)*(LK_CEIL+1);

    double g0[GSIZE];

    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;
    double gout[9*nfi*nfj*nfk] = {0};

    const int as_ish = envs.ibase ? ish: jsh;
    const int as_jsh = envs.ibase ? jsh: ish;

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e<LI, LJ, LK_CEIL>(envs, g0, norm, as_ish, as_jsh, ksh, ij, kl);
        const double ak2 = -2.0*c_bpcache.a1[kl];
        GINTgout3c2e_ipip2<LI,LJ,LK,NROOTS>(envs, gout, g0, ak2);
    } }
    GINTwrite_int3c2e_ipip(eri, gout, as_ish, as_jsh, ksh);
}

__device__
static void GINTwrite_int3c2e_ipip2_direct(GINTEnvVars envs, ERITensor eri,
    double* __restrict__ g0, const double ak2,
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
    const int nrys_roots = envs.nrys_roots;

    for (int tx = threadIdx.x; tx < (k1-k0)*(j1-j0)*(i1-i0); tx += blockDim.x) {
        const int k = tx / ((j1-j0)*(i1-i0));
        const int j = (tx / (i1-i0)) % (j1-j0);
        const int i = tx % (i1-i0);

        const int loc_k = c_l_locs[lk] + k;
        const int loc_j = c_l_locs[lj] + j;
        const int loc_i = c_l_locs[li] + i;

        int ix = dk * c_idx[loc_k] + dj * c_idx[loc_j] + di * c_idx[loc_i];
        int iy = dk * c_idy[loc_k] + dj * c_idy[loc_j] + di * c_idy[loc_i] + g_size;
        int iz = dk * c_idz[loc_k] + dj * c_idz[loc_j] + di * c_idz[loc_i] + g_size * 2;

        const int k_idx = c_idx[loc_k];
        const int k_idy = c_idy[loc_k];
        const int k_idz = c_idz[loc_k];

        double eri_xx = 0;
        double eri_xy = 0;
        double eri_xz = 0;
        double eri_yx = 0;
        double eri_yy = 0;
        double eri_yz = 0;
        double eri_zx = 0;
        double eri_zy = 0;
        double eri_zz = 0;
        for (int ir = 0; ir < nrys_roots; ++ir, ++ix, ++iy, ++iz){
            const double g0_x = g0[ix];
            const double g0_y = g0[iy];
            const double g0_z = g0[iz];

            // g2
            double g2_x = ak2*g0[ix+dk];
            double g2_y = ak2*g0[iy+dk];
            double g2_z = ak2*g0[iz+dk];
            g2_x += k_idx>0 ? k_idx*g0[ix-dk] : 0.0;
            g2_y += k_idy>0 ? k_idy*g0[iy-dk] : 0.0;
            g2_z += k_idz>0 ? k_idz*g0[iz-dk] : 0.0;

            // g3
            double g3_x = ak2*g0[ix+2*dk];
            double g3_y = ak2*g0[iy+2*dk];
            double g3_z = ak2*g0[iz+2*dk];
            g3_x += (k_idx+1)*g0[ix];
            g3_y += (k_idy+1)*g0[iy];
            g3_z += (k_idz+1)*g0[iz];
            g3_x *= ak2;
            g3_y *= ak2;
            g3_z *= ak2;
            if (k_idx > 0) { g3_x += ak2 * k_idx * g0[ix]; }
            if (k_idy > 0) { g3_y += ak2 * k_idy * g0[iy]; }
            if (k_idz > 0) { g3_z += ak2 * k_idz * g0[iz]; }
            if (k_idx > 1) { g3_x += k_idx * (k_idx-1) * g0[ix-2*dk]; }
            if (k_idy > 1) { g3_y += k_idy * (k_idy-1) * g0[iy-2*dk]; }
            if (k_idz > 1) { g3_z += k_idz * (k_idz-1) * g0[iz-2*dk]; }

            eri_xx += g3_x * g0_y * g0_z;
            eri_xy += g2_x * g2_y * g0_z;
            eri_xz += g2_x * g0_y * g2_z;
            eri_yx += g2_x * g2_y * g0_z;
            eri_yy += g0_x * g3_y * g0_z;
            eri_yz += g0_x * g2_y * g2_z;
            eri_zx += g2_x * g0_y * g2_z;
            eri_zy += g0_x * g2_y * g2_z;
            eri_zz += g0_x * g0_y * g3_z;
        }
        const int off = (i+i0) + jstride*(j+j0) + kstride*(k+k0);
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
    }
}

// General version
__global__
void GINTfill_int3c2e_ipip2_general_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    const int task_ij = blockIdx.x;// * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y;// * blockDim.y + threadIdx.y;

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

    extern __shared__ double g0[];

    const int as_ish = envs.ibase ? ish: jsh;
    const int as_jsh = envs.ibase ? jsh: ish;

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e_shared(envs, g0, norm, as_ish, as_jsh, ksh, ij, kl);
        const double ak2 = -2.0*c_bpcache.a1[kl];
        GINTwrite_int3c2e_ipip2_direct(envs, eri, g0, ak2, ish, jsh, ksh);
    } }
}


__global__
static void GINTfill_int3c2e_ipip2_kernel000(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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

    double gxx = 0;
    double gxy = 0;
    double gxz = 0;
    double gyx = 0;
    double gyy = 0;
    double gyz = 0;
    double gzx = 0;
    double gzy = 0;
    double gzz = 0;

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
            const double tmp3 = tmp1 * aij;
            const double b01 = b00 + tmp4 * aij;
            const double c0px = tmp3 * xijxkl;
            const double c0py = tmp3 * yijykl;
            const double c0pz = tmp3 * zijzkl;
            const double g_0 = 1;
            const double g_1 = c0px;
            const double g_2 = c0px * c0px + b01;
            const double g_3 = 1;
            const double g_4 = c0py;
            const double g_5 = c0py * c0py + b01;
            const double g_6 = weight0 * fac;
            const double g_7 = c0pz * g_6;
            const double g_8 = b01 * g_6 + c0pz * g_7;

            const double dgx_0 =       ak2 * g_1;
            const double dgx_1 = g_0 + ak2 * g_2;
            const double dgy_0 =       ak2 * g_4;
            const double dgy_1 = g_3 + ak2 * g_5;
            const double dgz_0 =       ak2 * g_7;
            const double dgz_1 = g_6 + ak2 * g_8;

            const double d2gx_0 = ak2 * dgx_1;
            const double d2gy_0 = ak2 * dgy_1;
            const double d2gz_0 = ak2 * dgz_1;

            gxx += d2gx_0  * g_3     * g_6;
            gxy += dgx_0   * dgy_0   * g_6;
            gxz += dgx_0   * g_3     * dgz_0;
            gyx += dgx_0   * dgy_0   * g_6;
            gyy += g_0     * d2gy_0  * g_6;
            gyz += g_0     * dgy_0   * dgz_0;
            gzx += dgx_0   * g_3     * dgz_0;
            gzy += g_0     * dgy_0   * dgz_0;
            gzz += g_0     * g_3     * d2gz_0;
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
