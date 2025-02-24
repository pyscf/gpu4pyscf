template <int NROOTS, int GSIZE> __attribute__((always_inline))
void GINTfill_int3c2e_ip1_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets, sycl::nd_item<2>& item)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = static_cast<int>( item.get_global_id(1) );
    int task_kl = static_cast<int>( item.get_global_id(0) );

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
    int *bas_pair2bra = c_bpcache.get().bas_pair2bra;
    int *bas_pair2ket = c_bpcache.get().bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double* __restrict__ exp = c_bpcache.get().a1;
    double g[2*GSIZE];
    double *f = g + GSIZE;

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
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_2e_2d4d<NROOTS>(envs, g, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            double ai2 = -2.0*exp[ij];
            GINTnabla1i_2e<NROOTS>(envs, f, g, ai2, envs.i_l, envs.j_l, envs.k_l);
            GINTwrite_int3c2e_ip_direct<NROOTS>(envs, eri, f, g, ish, jsh, ksh);
    } }
}
