#!/bin/sh
gen_nabla_script(){
    if [ ${IP_TYPE} = 'pass1' ]; then
        GET_JK_DIRECT="GINTkernel_int3c2e_getj_pass1<${ROOT}>(envs, jk, g, ish, jsh, ksh);"

    fi
    if [ ${IP_TYPE} = 'pass2' ]; then
        GET_JK_DIRECT="GINTkernel_int3c2e_getj_pass2<${ROOT}>(envs, jk, g, ish, jsh, ksh);"
    fi
}

gen_code(){
    FILE='g3c2e_'${IP_TYPE}'.cu'
    cat /dev/null > ${FILE}
    # header 
    echo '/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
 ' >> ${FILE}
    echo "${TEMPLATE}" >> ${FILE}
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
    
    for i in 2 3 4 5 6 7 8 9
    do
        ROOT=$i
        ROOT2=$((2*i))
        ROOT_1=$((i-1))
        GS='GSIZE'${ROOT}'_INT3C'
        gen_nabla_script
        echo "
#if POLYFIT_ORDER_IP >= ${ROOT}
template <> __global__
void GINTint3c2e_${IP_TYPE}_j_kernel<${ROOT},${GS}>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    ${BLK0}
    double uw[${ROOT2}];
    double g[${GS}];
    
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
        double aij = a12[ij];
        double xij = x12[ij];
        double yij = y12[ij];
        double zij = z12[ij];
        for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {  
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
            double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
            GINTrys_root${ROOT}(x, uw);
            GINTg0_2e_2d4d<${ROOT}>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            ${GET_JK_DIRECT}
    } }
}
#endif " >> ${FILE}
    done
}


### main script
IP_TYPE='pass1'
TEMPLATE="
template <int NROOTS, int GSIZE> __global__
static void GINTint3c2e_pass1_j_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets){
    return;
}"
gen_code

IP_TYPE='pass2'
TEMPLATE="
template <int NROOTS, int GSIZE> __global__
static void GINTint3c2e_pass2_j_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets){
    return;
}
"
gen_code
