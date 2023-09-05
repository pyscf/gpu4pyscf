#!/bin/sh
gen_nabla_script(){
    if [ ${IP_TYPE} = 'ip1' ]; then
        TEMPLATE="
template <int NROOTS, int GSIZE> __global__
static void GINTint3c2e_ip1_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets){
    return;
}"
        NABLA_SCRIPT="
            double ai2 = -2.0*exp[ij];
            GINTnabla1i_2e<${ROOT}>(envs, f, g, ai2, envs.i_l, envs.j_l, envs.k_l);"
        GET_JK='GINTkernel_int3c2e_ip1_getjk(envs, jk, gout, ish, jsh, ksh);'
        GET_JK_DIRECT="GINTkernel_int3c2e_ip1_getjk_direct<${ROOT}>(envs, jk, j3, k3, f, g, ish, jsh, ksh);"
        WRITE_JK='write_int3c2e_ip1_jk(jk, j3, k3, ish);'

    fi
    if [ ${IP_TYPE} = 'ip2' ]; then
        NABLA_SCRIPT="
            double ak2 = -2.0*exp[kl];
            GINTnabla1k_2e<${ROOT}>(envs, f, g, ak2, envs.i_l, envs.j_l, envs.k_l);"
        GET_JK='GINTkernel_int3c2e_ip2_getjk(envs, jk, gout, ish, jsh, ksh);'
        GET_JK_DIRECT="GINTkernel_int3c2e_ip2_getjk_direct<${ROOT}>(envs, jk, j3, k3, f, g, ish, jsh, ksh);"
        WRITE_JK='write_int3c2e_ip2_jk(jk, j3, k3, ksh);'
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
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
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

    BLK1='            
            double aij = a12[ij];
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
        ROOT=$i
        ROOT2=$((2*i))
        ROOT_1=$((i-1))
        GS='GSIZE'${ROOT}'_INT3C'
        GO='GOUTSIZE'${ROOT}'_INT3C'
        GO_1='GOUTSIZE'${ROOT_1}'_INT3C'
        gen_nabla_script
        echo "
#if POLYFIT_ORDER_IP >= ${ROOT}
template <> __global__
void GINTint3c2e_${IP_TYPE}_jk_kernel<${ROOT},${GS}>(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    ${BLK0}
    double* __restrict__ exp = c_bpcache.a1;
    double uw[${ROOT2}];
    double g[2*${GS}];
    double *f = g + ${GS};
    
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
    
    double j3[GPU_CART_MAX * 3];
    double k3[GPU_CART_MAX * 3];
    for (int k = 0; k < GPU_CART_MAX * 3; k++){
        j3[k] = 0.0;
        k3[k] = 0.0;
    }
    if (active) {
        for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
                ${BLK1}
                GINTrys_root${ROOT}(x, uw);
                GINTscale_u<${ROOT}>(uw, theta);
                GINTg0_2e_2d4d<${ROOT}>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
                ${NABLA_SCRIPT}
                ${GET_JK_DIRECT}
            } 
        }
    }
    
    ${WRITE_JK}
}
#endif " >> ${FILE}
    done
}


### main script
IP_TYPE='ip1'
TEMPLATE="
template <int NROOTS, int GSIZE> __global__
static void GINTint3c2e_ip1_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets){
    return;
}"
gen_code

IP_TYPE='ip2'
TEMPLATE="
template <int NROOTS, int GSIZE> __global__
static void GINTint3c2e_ip2_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets){
    return;
}
"
gen_code
