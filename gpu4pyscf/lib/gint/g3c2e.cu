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


__device__
static void GINTwrite_int3c2e_direct(GINTEnvVars envs, ERITensor eri, double* g, 
    const int ish, const int jsh, const int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;

    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;

    const int li = envs.i_l;
    const int lj = envs.j_l;
    const int lk = envs.k_l;
    const int nrys_roots = envs.nrys_roots;

    //for (int k = k0; k < k1; ++k) {
    //for (int j = j0; j < j1; ++j) {
    //for (int i = i0; i < i1; ++i) {
    for (int tx = threadIdx.x; tx < (k1-k0)*(j1-j0)*(i1-i0); tx += blockDim.x) {
        const int k = tx / ((j1-j0)*(i1-i0));
        const int j = (tx / (i1-i0)) % (j1-j0);
        const int i = tx % (i1-i0);

        const int loc_k = c_l_locs[lk] + k;
        const int loc_j = c_l_locs[lj] + j;
        const int loc_i = c_l_locs[li] + i;

        const int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
        const int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
        const int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

        double eri_data = 0;
#pragma unroll
        for (int ir = 0; ir < nrys_roots; ++ir){
            eri_data += g[ix + ir] * g[iy + ir] * g[iz + ir];
        }
        const int idx = (i+i0) + jstride*(j+j0) + kstride*(k+k0);
        eri.data[idx] += eri_data;
    }
}

__device__
static void GINTmemset_int3c2e(GINTEnvVars envs, ERITensor eri, int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    size_t jstride = eri.stride_j;
    size_t kstride = eri.stride_k;
    int i0 = ao_loc[ish  ] - eri.ao_offsets_i;
    int i1 = ao_loc[ish+1] - eri.ao_offsets_i;
    int j0 = ao_loc[jsh  ] - eri.ao_offsets_j;
    int j1 = ao_loc[jsh+1] - eri.ao_offsets_j;
    int k0 = ao_loc[ksh  ] - eri.ao_offsets_k;
    int k1 = ao_loc[ksh+1] - eri.ao_offsets_k;

    for (int tx = threadIdx.x; tx < (k1-k0)*(j1-j0)*(i1-i0); tx += blockDim.x) {
        const int k = tx / ((j1-j0)*(i1-i0));
        const int j = (tx / (i1-i0)) % (j1-j0);
        const int i = tx % (i1-i0);

        const int idx = (i+i0) + jstride*(j+j0) + kstride*(k+k0);
        eri.data[idx] = 0.0;
    }
}

__global__
void GINTfill_int3c2e_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    const int task_ij = blockIdx.x;// * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y;// * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }
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

    GINTmemset_int3c2e(envs, eri, ish, jsh, ksh);
    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e_shared(envs, g, as_ish, as_jsh, ksh, ij, kl);
        GINTwrite_int3c2e_direct(envs, eri, g, ish, jsh, ksh);
    } }
}

__global__
static void GINTfill_int3c2e_kernel0000(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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
    
    double gout0 = 0;
    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
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
        double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
        if (x > 3.e-7) {
            const double tt = sqrt(x);
            const double fmt0 = SQRTPIE4 / tt * erf(tt);
            fac *= fmt0;
        }
        gout0 += fac;
    } }
    
    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;

    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh] - eri.ao_offsets_k;
    eri.data[k0*kstride+j0*jstride+i0] = gout0;
}

__global__
static void GINTfill_int3c2e_kernel0010(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;

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
        //const double tmp4 = .5 / (u2 * aijkl + a1);
        //const double b00 = u2 * tmp4;
        //const double tmp1 = aij * u2 / (u2 * aijkl + a1);;
        const double tmp3 = aij * u2 / (u2 * aijkl + a1);;
        const double c0px = tmp3 * xijxkl;
        const double c0py = tmp3 * yijykl;
        const double c0pz = tmp3 * zijzkl;
        const double g_0 = 1;
        const double g_1 = c0px;
        const double g_2 = 1;
        const double g_3 = c0py;
        const double g_4 = weight0 * fac;
        const double g_5 = c0pz * g_4;
        gout0 += g_1 * g_2 * g_4;
        gout1 += g_0 * g_3 * g_4;
        gout2 += g_0 * g_2 * g_5;
    } }

    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh] - eri.ao_offsets_k;
    double* __restrict__ eri_ij = eri.data + k0*kstride+j0*jstride+i0;

    eri_ij[0] = gout0;
    eri_ij[1*kstride] = gout1;
    eri_ij[2*kstride] = gout2;
}

__global__
static void GINTfill_int3c2e_kernel1000(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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

    const int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    const double xi = bas_x[ish];
    const double yi = bas_y[ish];
    const double zi = bas_z[ish];
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
        const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
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
        const double tmp2 = akl * u2 / (u2 * aijkl + a1);;
        const double c00x = xij - xi - tmp2 * xijxkl;
        const double c00y = yij - yi - tmp2 * yijykl;
        const double c00z = zij - zi - tmp2 * zijzkl;
        const double g_0 = 1;
        const double g_1 = c00x;
        const double g_2 = 1;
        const double g_3 = c00y;
        const double g_4 = norm * fac * weight0;
        const double g_5 = g_4 * c00z;
        gout0 += g_1 * g_2 * g_4;
        gout1 += g_0 * g_3 * g_4;
        gout2 += g_0 * g_2 * g_5;
    } }

    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh] - eri.ao_offsets_k;
   
    double* __restrict__ eri_ij = eri.data + k0*kstride+j0*jstride+i0;

    eri_ij[0] = gout0;
    eri_ij[1] = gout1;
    eri_ij[2] = gout2;
}

__global__
static void GINTfill_int3c2e_kernel0100(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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

    const int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    const double xj = bas_x[jsh];
    const double yj = bas_y[jsh];
    const double zj = bas_z[jsh];
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
        const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
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
        const double tmp2 = akl * u2 / (u2 * aijkl + a1);;
        const double c00x = xij - xj - tmp2 * xijxkl;
        const double c00y = yij - yj - tmp2 * yijykl;
        const double c00z = zij - zj - tmp2 * zijzkl;
        const double g_0 = 1;
        const double g_1 = c00x;
        const double g_2 = 1;
        const double g_3 = c00y;
        const double g_4 = norm * fac * weight0;
        const double g_5 = g_4 * c00z;
        gout0 += g_1 * g_2 * g_4;
        gout1 += g_0 * g_3 * g_4;
        gout2 += g_0 * g_2 * g_5;
    } }

    const size_t jstride = eri.stride_j;
    const size_t kstride = eri.stride_k;
    
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - eri.ao_offsets_i;
    const int j0 = ao_loc[jsh] - eri.ao_offsets_j;
    const int k0 = ao_loc[ksh] - eri.ao_offsets_k;
   
    double* __restrict__ eri_ij = eri.data + k0*kstride+j0*jstride+i0;

    eri_ij[0] = gout0;
    eri_ij[1*jstride] = gout1;
    eri_ij[2*jstride] = gout2;
}
