#!/bin/sh

gen_nabla_script(){
    if [ ${IP_TYPE} = 'ipip1' ];
    then
        DEF_EXP="double* __restrict__ exp_bra = c_bpcache.a1;"
        NABLA_SCRIPT="
            double ai2 = -2.0*exp_bra[ij];
            GINTnabla1i_2e<${ROOT}>(envs, g1, g0, ai2, envs.i_l+1, envs.j_l, envs.k_l);
            GINTnabla1i_2e<${ROOT}>(envs, g2, g0, ai2, envs.i_l,   envs.j_l, envs.k_l);
            GINTnabla1i_2e<${ROOT}>(envs, g3, g1, ai2, envs.i_l,   envs.j_l, envs.k_l);"
    fi

    if [ ${IP_TYPE} = 'ipip2' ];
    then
        DEF_EXP="double* __restrict__ exp_bra = c_bpcache.a1;"
        NABLA_SCRIPT="
            double ak2 = -2.0*exp_bra[kl];
            GINTnabla1k_2e<${ROOT}>(envs, g1, g0, ak2, envs.i_l,   envs.j_l, envs.k_l+1);
            GINTnabla1k_2e<${ROOT}>(envs, g2, g0, ak2, envs.i_l,   envs.j_l, envs.k_l);
            GINTnabla1k_2e<${ROOT}>(envs, g3, g1, ak2, envs.i_l,   envs.j_l, envs.k_l);"
    fi

    if [ ${IP_TYPE} = 'ip1ip2' ];
    then
        DEF_EXP="double* __restrict__ exp_bra = c_bpcache.a1;"
        NABLA_SCRIPT="
            double ai2 = -2.0*exp_bra[ij];
            double ak2 = -2.0*exp_bra[kl];
            GINTnabla1k_2e<${ROOT}>(envs, g1, g0, ak2, envs.i_l+1, envs.j_l, envs.k_l);
            GINTnabla1i_2e<${ROOT}>(envs, g2, g0, ai2, envs.i_l,   envs.j_l, envs.k_l);
            GINTnabla1i_2e<${ROOT}>(envs, g3, g1, ai2, envs.i_l,   envs.j_l, envs.k_l);"
    fi

    if [ ${IP_TYPE} = 'ipvip1' ];
    then
        DEF_EXP="double* __restrict__ exp_bra = c_bpcache.a1;
    double* __restrict__ exp_ket = c_bpcache.a2;"
        NABLA_SCRIPT="
            double ai2 = -2.0*exp_bra[ij];
            double aj2 = -2.0*exp_ket[ij];
            GINTnabla1j_2e<${ROOT}>(envs, g1, g0, aj2, envs.i_l+1, envs.j_l, envs.k_l);
            GINTnabla1i_2e<${ROOT}>(envs, g2, g0, ai2, envs.i_l,   envs.j_l, envs.k_l);
            GINTnabla1i_2e<${ROOT}>(envs, g3, g1, ai2, envs.i_l,   envs.j_l, envs.k_l);"
    fi
}

gen_code(){
    FILE='g3c2e_'${IP_TYPE}'.cu'
    cat /dev/null > ${FILE}
    # header 
    echo '
// general case, it is not supposed to be used in actual execution
template <int NROOTS, int GSIZE> __global__
static void GINTfill_int3c2e_'${IP_TYPE}'_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    fprintf(stderr, "general function is not implemented");
}' >> ${FILE}
    BLK0='int ntasks_ij = offsets.ntasks_ij;
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
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];'
    
    BLK1='double aij = a12[ij];
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
            double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);'

    for i in 2 3 4 5 6 7 8 9
    do 
        N=$i
        N2=$((2*i))
        N_1=$((i-1))
        ROOT=$i
        GS='GSIZE'${N}'_INT3C'
        GO='GOUTSIZE'${N_1}'_INT3C'
        gen_nabla_script
        echo "
#if POLYFIT_ORDER_IP >= ${N}
template <> __global__
void GINTfill_int3c2e_${IP_TYPE}_kernel<${N},${GS}>(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    ${BLK0}
    double uw[${N2}];
    
    double g0[4*${GS}];
    double *g1 = g0 + ${GS};
    double *g2 = g1 + ${GS};
    double *g3 = g2 + ${GS};
    ${DEF_EXP}
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
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            ${BLK1}
            GINTrys_root${N}(x, uw);
            GINTscale_u<${ROOT}>(uw, theta);
            GINTg0_2e_2d4d<${N}>(envs, g0, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            ${NABLA_SCRIPT}
            GINTwrite_int3c2e_ipip_direct<${N}>(envs, eri, g0, g1, g2, g3, ish, jsh, ksh);
    } }
}
#endif " >> ${FILE}
    done
}

### main script

IP_TYPE='ipip1'
gen_code

IP_TYPE='ipip2'
gen_code

IP_TYPE='ip1ip2'
gen_code

IP_TYPE='ipvip1'
gen_code
