/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

template <int orderi, int orderj, int LI, int LJ> __device__
void type1_cart_unrolled_kernel(double *gctr,
                const int ish, const int jsh, const int ksh,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];
    const double r2ca = rca[0]*rca[0] + rca[1]*rca[1] + rca[2]*rca[2];
    const double r2cb = rcb[0]*rcb[0] + rcb[1]*rcb[1] + rcb[2]*rcb[2];

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    constexpr int LIJ1 = LI+LJ+1;
    constexpr int LIJ3 = LIJ1*LIJ1*LIJ1;

    __shared__ double rad_ang[LIJ3];
    set_shared_memory(rad_ang, LIJ3);

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];
    for (int ip = 0; ip < npi; ip++){
        for (int jp = 0; jp < npj; jp++){
            double rij[3];
            double ai_prim = ai[ip];
            double aj_prim = aj[jp];
            rij[0] = ai_prim * rca[0] + aj_prim * rcb[0];
            rij[1] = ai_prim * rca[1] + aj_prim * rcb[1];
            rij[2] = ai_prim * rca[2] + aj_prim * rcb[2];
            const double k = 2.0 * norm3d(rij[0], rij[1], rij[2]);
            const double aij = ai_prim + aj_prim;

            __shared__ double rad_all[LIJ1*LIJ1];
            type1_rad_part(rad_all, LI+LJ, k, aij, ur);
            __syncthreads();

            const double eij = exp(-ai_prim*r2ca - aj_prim*r2cb);
            const double eaij = eij * pow(-2.0*ai_prim, orderi) * pow(-2.0*aj_prim, orderj);
            const double ceij = eaij * ci[ip] * cj[jp];
            type1_rad_ang<LI+LJ>(rad_ang, rij, rad_all, fac*ceij);
            //type1_rad_ang(rad_ang, LI+LJ, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }

    constexpr int NFI = (LI+1)*(LI+2)/2;
    constexpr int NFJ = (LJ+1)*(LJ+2)/2;
    double fi[3*NFI];
    cache_fac<LI>(fi, rca);
    double fj[3*NFJ];
    cache_fac<LJ>(fj, rcb);

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int mi = ij%nfi;
        const int mj = ij/nfi;

        const int iy = _cart_pow_y[mi];
        const int iz = _cart_pow_z[mi];
        const int ix = LI - iy - iz;
        const int ix_off = (ix+1)*ix/2;
        const int iy_off = (iy+1)*iy/2 + nfi;
        const int iz_off = (iz+1)*iz/2 + 2*nfi;

        const int jy = _cart_pow_y[mj];
        const int jz = _cart_pow_z[mj];
        const int jx = LJ - jy - jz;
        const int jx_off = (jx+1)*jx/2;
        const int jy_off = (jy+1)*jy/2 + nfj;
        const int jz_off = (jz+1)*jz/2 + 2*nfj;

        // cache ifac and jfac in register
        double tmp = 0.0;
        for (int i1 = 0; i1 <= ix; i1++){
        for (int i2 = 0; i2 <= iy; i2++){
        for (int i3 = 0; i3 <= iz; i3++){
            const double ifac = fi[i1+ix_off] * fi[i2+iy_off] * fi[i3+iz_off];
            for (int j1 = 0; j1 <= jx; j1++){
            for (int j2 = 0; j2 <= jy; j2++){
            for (int j3 = 0; j3 <= jz; j3++){
                const double jfac = fj[j1+jx_off] * fj[j2+jy_off] * fj[j3+jz_off];
                const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
                tmp += ifac * jfac * rad_ang[ijr];
            }}}
        }}}
        gctr[ij] = tmp;
    }
}

template <int orderi, int orderj> __device__
void type1_cart_kernel(double *gctr,
                const int LI, const int LJ,
                const int ish, const int jsh, const int ksh,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    extern __shared__ double smem[];

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];
    const double r2ca = rca[0]*rca[0] + rca[1]*rca[1] + rca[2]*rca[2];
    const double r2cb = rcb[0]*rcb[0] + rcb[1]*rcb[1] + rcb[2]*rcb[2];

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    const int LIJ1 = LI+LJ+1;
    const int LIJ3 = LIJ1*LIJ1*LIJ1;

    double *rad_ang = smem;
    for (int i = threadIdx.x; i < LIJ3; i+=blockDim.x) {
        rad_ang[i] = 0;
    }
    __syncthreads();

    double *rad_all = rad_ang + LIJ3;
    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];
    for (int ip = 0; ip < npi; ip++){
        for (int jp = 0; jp < npj; jp++){
            double rij[3];
            const double ai_prim = ai[ip];
            const double aj_prim = aj[jp];
            rij[0] = ai_prim * rca[0] + aj_prim * rcb[0];
            rij[1] = ai_prim * rca[1] + aj_prim * rcb[1];
            rij[2] = ai_prim * rca[2] + aj_prim * rcb[2];
            const double k = 2.0 * norm3d(rij[0], rij[1], rij[2]);
            const double aij = ai_prim + aj_prim;
            type1_rad_part(rad_all, LI+LJ, k, aij, ur);
            __syncthreads();

            const double eij = exp(-ai_prim*r2ca - aj_prim*r2cb);
            const double eaij = eij * pow(-2.0*ai_prim, orderi) * pow(-2.0*aj_prim, orderj);
            const double ceij = eaij * ci[ip] * cj[jp];
            type1_rad_ang(rad_ang, LI+LJ, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }

    constexpr int NFI_MAX = (AO_LMAX+orderi+1)*(AO_LMAX+orderi+2)/2;
    constexpr int NFJ_MAX = (AO_LMAX+orderj+1)*(AO_LMAX+orderj+2)/2;
    double fi[3*NFI_MAX];
    cache_fac(fi, LI, rca);
    double fj[3*NFJ_MAX];
    cache_fac(fj, LJ, rcb);

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int mi = ij%nfi;
        const int mj = ij/nfi;

        const int iy = _cart_pow_y[mi];
        const int iz = _cart_pow_z[mi];
        const int ix = LI - iy - iz;
        const int ix_off = (ix+1)*ix/2;
        const int iy_off = (iy+1)*iy/2 + nfi;
        const int iz_off = (iz+1)*iz/2 + 2*nfi;

        const int jy = _cart_pow_y[mj];
        const int jz = _cart_pow_z[mj];
        const int jx = LJ - jy - jz;
        const int jx_off = (jx+1)*jx/2;
        const int jy_off = (jy+1)*jy/2 + nfj;
        const int jz_off = (jz+1)*jz/2 + 2*nfj;

        // cache ifac and jfac in register
        double tmp = 0.0;
        for (int i1 = 0; i1 <= ix; i1++){
        for (int i2 = 0; i2 <= iy; i2++){
        for (int i3 = 0; i3 <= iz; i3++){
            const double ifac = fi[i1+ix_off] * fi[i2+iy_off] * fi[i3+iz_off];
            for (int j1 = 0; j1 <= jx; j1++){
            for (int j2 = 0; j2 <= jy; j2++){
            for (int j3 = 0; j3 <= jz; j3++){
                const double jfac = fj[j1+jx_off] * fj[j2+jy_off] * fj[j3+jz_off];
                const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
                tmp += ifac * jfac * rad_ang[ijr];
            }}}
        }}}
        gctr[ij] = tmp;
    }
}

// Unrolling case
template <int LI, int LJ> __global__
void type1_cart_ip1(double *gctr,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += 3*ecp_id*nao*nao + ioff*nao + joff;

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    __shared__ double gctr_smem[nfi*nfj*3];
    for (int ij = threadIdx.x; ij < nfi*nfj*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int nfi1 = (LI+2)*(LI+3)/2;
    __shared__ double buf[nfi1*nfj];

    type1_cart_unrolled_kernel<1,0,LI+1,LJ>(
        buf, ish, jsh, ksh,
        ecpbas, ecploc,
        atm, bas, env);
    __syncthreads();
    _li_down(gctr_smem, buf, LI, LJ);
    __syncthreads();

    if constexpr (LI > 0){
        type1_cart_unrolled_kernel<0,0,LI-1,LJ>(
            buf, ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env);
        __syncthreads();
        _li_up(gctr_smem, buf, LI, LJ);
        __syncthreads();
    }

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gctr +   nao*nao;
        double *gz = gctr + 2*nao*nao;
        atomicAdd(gx+i*nao+j, gctr_smem[ij]);
        atomicAdd(gy+i*nao+j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz+i*nao+j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}

__global__
void type1_cart_ip1_general(double *gctr,
                const int LI, const int LJ,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += 3*ecp_id*nao*nao + ioff*nao + joff;

    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];
    for (int ij = threadIdx.x; ij < NF_MAX*NF_MAX*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int nfi_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf[nfi_max*nfj_max];

    type1_cart_kernel<1,0>(
        buf, LI+1, LJ,
        ish, jsh, ksh,
        ecpbas, ecploc,
        atm, bas, env);
    __syncthreads();
    _li_down(gctr_smem, buf, LI, LJ);
    __syncthreads();

    if (LI > 0){
        type1_cart_kernel<0,0>(
            buf, LI-1, LJ,
            ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env);
        __syncthreads();
        _li_up(gctr_smem, buf, LI, LJ);
        __syncthreads();
    }

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gctr +   nao*nao;
        double *gz = gctr + 2*nao*nao;
        atomicAdd(gx+i*nao+j, gctr_smem[ij]);
        atomicAdd(gy+i*nao+j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz+i*nao+j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}
