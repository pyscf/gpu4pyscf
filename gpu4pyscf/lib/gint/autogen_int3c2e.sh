#!/bin/sh

FILE='g3c2e.cu'
GOUT_FUN='GINTgout3c2e'
WRITE_FUN='GINTwrite_int3c2e'

cat /dev/null > ${FILE}

# header 
echo '
// general case, it is not supposed to be used in actual execution
template <int NROOTS, int GSIZE> __global__
static void GINTfill_int3c2e_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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
    GO='GOUTSIZE'${ROOT}'_INT3C'
    GS='GSIZE'${ROOT}'_INT3C'
    echo "
#if POLYFIT_ORDER_IP >= ${ROOT}
template <> __global__
void GINTfill_int3c2e_kernel<${ROOT},${GS}>(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
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
        for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            ${BLK1}
            GINTrys_root${ROOT}(x, uw);
            GINTscale_u<${ROOT}>(uw, theta);
            GINTg0_2e_2d4d<${ROOT}>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            GINTwrite_int3c2e_direct<${i}>(envs, eri, g, ish, jsh, ksh);
    } }
}

#endif " >> ${FILE}
done