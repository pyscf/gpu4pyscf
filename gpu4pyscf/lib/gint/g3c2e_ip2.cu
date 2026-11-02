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
static void GINTgout3c2e_ip(GINTEnvVars envs, double* __restrict__ gout, double* __restrict__ g, const double ak2)
{
    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

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
        
        const int k_idx = idx[loc_k];
        const int k_idy = idy[loc_k];
        const int k_idz = idz[loc_k];

        int ix = dk * k_idx + dj * idx[loc_j] + di * idx[loc_i];
        int iy = dk * k_idy + dj * idy[loc_j] + di * idy[loc_i] + g_size;
        int iz = dk * k_idz + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

#pragma unroll
        for (int n = 0; n < NROOTS; ++n, ++ix, ++iy, ++iz) {
            const double gx = g[ix];
            const double gy = g[iy];
            const double gz = g[iz];

            double fx = ak2*g[ix+dk];
            double fy = ak2*g[iy+dk];
            double fz = ak2*g[iz+dk];

            fx += k_idx>0 ? k_idx*g[ix-dk] : 0.0;
            fy += k_idy>0 ? k_idy*g[iy-dk] : 0.0;
            fz += k_idz>0 ? k_idz*g[iz-dk] : 0.0;

            gout[3*i + 0] += fx * gy * gz;
            gout[3*i + 1] += gx * fy * gz;
            gout[3*i + 2] += gx * gy * fz;
        }
    }}}
}

// Unrolled version
template <int LI, int LJ, int LK> __global__
void GINTfill_int3c2e_ip2_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    const int *bas_pair2bra = c_bpcache.bas_pair2bra;
    const int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];

    constexpr int LK_CEIL = LK + 1;
    constexpr int NROOTS = (LI+LJ+LK_CEIL)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI+1)*(LJ+1)*(LK_CEIL+1);

    double g[GSIZE];

    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;
    double gout[3*nfi*nfj*nfk] = {0};

    const int as_ish = envs.ibase ? ish: jsh;
    const int as_jsh = envs.ibase ? jsh: ish;

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e<LI, LJ, LK_CEIL>(envs, g, as_ish, as_jsh, ksh, ij, kl);
        const double ak2 = -2.0*c_bpcache.a1[kl];
        GINTgout3c2e_ip<LI,LJ,LK,NROOTS>(envs, gout, g, ak2);
    } }
    GINTwrite_int3c2e_ip(eri, gout, as_ish, as_jsh, ksh);
}

__device__
static void GINTwrite_int3c2e_ip2_direct(GINTEnvVars envs, ERITensor eri, double* g, double ak2, int ish, int jsh, int ksh)
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

        double eri_x = 0;
        double eri_y = 0;
        double eri_z = 0;
#pragma unroll
        for (int ir = 0; ir < nrys_roots; ++ir, ++ix, ++iy, ++iz){
            const double gx = g[ix];
            const double gy = g[iy];
            const double gz = g[iz];

            double fx = ak2*g[ix+dk];
            double fy = ak2*g[iy+dk];
            double fz = ak2*g[iz+dk];

            fx += k_idx>0 ? k_idx*g[ix-dk] : 0.0;
            fy += k_idy>0 ? k_idy*g[iy-dk] : 0.0;
            fz += k_idz>0 ? k_idz*g[iz-dk] : 0.0;

            eri_x += fx * gy * gz;
            eri_y += gx * fy * gz;
            eri_z += gx * gy * fz;
        }

        const int off = (i+i0) + jstride*(j+j0) + (k+k0)*kstride;
        double *eri_data = eri.data + off;
        eri_data[0*lstride] += eri_x;
        eri_data[1*lstride] += eri_y;
        eri_data[2*lstride] += eri_z;
    }
}


// General version
__global__
void GINTfill_int3c2e_ip2_general_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    const int task_ij = blockIdx.x;// * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y;// * blockDim.y + threadIdx.y;
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

    extern __shared__ double g[];

    const int as_ish = envs.ibase ? ish: jsh;
    const int as_jsh = envs.ibase ? jsh: ish;

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e_shared(envs, g, as_ish, as_jsh, ksh, ij, kl);
        const double ak2 = -2.0*c_bpcache.a1[kl];
        GINTwrite_int3c2e_ip2_direct(envs, eri, g, ak2, ish, jsh, ksh);
    } }
}

__global__
static void GINTfill_int3c2e_ip2_kernel000(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double* __restrict__ a1 = c_bpcache.a1;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;

    const int prim_ij0 = prim_ij;
    const int prim_ij1 = prim_ij + nprim_ij;
    const int prim_kl0 = prim_kl;
    const int prim_kl1 = prim_kl + nprim_kl;
    for (int ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (int kl = prim_kl0; kl < prim_kl1; ++kl) {
        const double ak2 = -2.0*a1[kl];
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
        double root0, weight0;
        if (x < 3.e-7) {
            root0 = 0.5;
            weight0 = 1.;
        } else {
            const double tt = sqrt(x);
            const double fmt0 = SQRTPIE4 / tt * erf(tt);
            weight0 = fmt0;
            const double e = exp(-x);
            const double b = .5 / x;
            const double fmt1 = b * (fmt0 - e);
            root0 = fmt1 / (fmt0 - fmt1);
        }
        root0 /= root0 + 1 - root0 * theta;
        const double u2 = a0 * root0;
        const double tmp1 = u2 / (u2 * aijkl + a1);
        const double tmp3 = tmp1 * aij;
        const double c0px = tmp3 * xijxkl;
        const double c0py = tmp3 * yijykl;
        const double c0pz = tmp3 * zijzkl;
        const double g_0 = 1;
        const double g_1 = c0px;
        const double g_2 = 1;
        const double g_3 = c0py;
        const double g_4 = weight0 * fac;
        const double g_5 = c0pz * g_4;

        const double f_1 = ak2 * g_1;
        const double f_3 = ak2 * g_3;
        const double f_5 = ak2 * g_5;

        gout0 += f_1 * g_2 * g_4;
        gout1 += g_0 * f_3 * g_4;
        gout2 += g_0 * g_2 * f_5;

    } }

    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    const size_t lstride = eri.stride_l;
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh] - eri.ao_offsets_k;

    double* __restrict__ eri_ij = eri.data + k0*kstride+j0*jstride+i0;

    eri_ij[0*lstride] = gout0;
    eri_ij[1*lstride] = gout1;
    eri_ij[2*lstride] = gout2;
}

