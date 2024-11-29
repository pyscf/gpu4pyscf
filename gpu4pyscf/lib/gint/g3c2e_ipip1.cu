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
void GINTfill_int3c2e_ipip1_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
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
    int lsh = bas_pair2ket[bas_kl];

    double* __restrict__ exp_bra = c_bpcache.a1;
    constexpr int LI_CEIL = LI + 2;
    constexpr int NROOTS = (LI_CEIL+LJ+LK)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI_CEIL+1)*(LJ+1)*(LK+1);
    
    double g0[4*GSIZE];
    double *g1 = g0 + GSIZE;
    double *g2 = g1 + GSIZE;
    double *g3 = g2 + GSIZE;

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
    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_int3c2e<LI_CEIL, LJ, LK>(envs, g0, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

            double ai2 = -2.0*exp_bra[ij];
            GINTnabla1i_2e<LI+1, LJ, LK, NROOTS>(envs, g1, g0, ai2);
            GINTnabla1i_2e<LI,   LJ, LK, NROOTS>(envs, g2, g0, ai2);
            GINTnabla1i_2e<LI,   LJ, LK, NROOTS>(envs, g3, g1, ai2);
            GINTwrite_int3c2e_ipip_direct<LI, LJ, LK>(envs, eri, g0, g1, g2, g3, ish, jsh, ksh);
    } }
}

// General version
template <int NROOTS, int GSIZE> __global__
void GINTfill_int3c2e_ipip1_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
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
    int lsh = bas_pair2ket[bas_kl];

    double g0[4*GSIZE];
    double *g1 = g0 + GSIZE;
    double *g2 = g1 + GSIZE;
    double *g3 = g2 + GSIZE;
    double* __restrict__ exp_bra = c_bpcache.a1;

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
    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_int3c2e<NROOTS>(envs, g0, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

            double ai2 = -2.0*exp_bra[ij];
            GINTnabla1i_2e<NROOTS>(envs, g1, g0, ai2, envs.i_l+1, envs.j_l, envs.k_l);
            GINTnabla1i_2e<NROOTS>(envs, g2, g0, ai2, envs.i_l,   envs.j_l, envs.k_l);
            GINTnabla1i_2e<NROOTS>(envs, g3, g1, ai2, envs.i_l,   envs.j_l, envs.k_l);
            GINTwrite_int3c2e_ipip_direct<NROOTS>(envs, eri, g0, g1, g2, g3, ish, jsh, ksh);
    } }
}
