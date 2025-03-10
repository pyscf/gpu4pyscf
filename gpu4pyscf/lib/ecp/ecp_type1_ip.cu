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


template <int orderi, int orderj> __device__
void type1_cart_kernel(double *gctr, 
                const int LI, const int LJ,
                int ish, int jsh, int ksh,
                const int *ecpbas, const int *ecploc, 
                const int *atm, const int *bas, const double *env)
{
    extern __shared__ double smem[];

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
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
            double ai_prim = ai[ip];
            double aj_prim = aj[jp];
            rij[0] = ai_prim * rca[0] + aj_prim * rcb[0];
            rij[1] = ai_prim * rca[1] + aj_prim * rcb[1];
            rij[2] = ai_prim * rca[2] + aj_prim * rcb[2];
            const double k = 2.0 * norm3d(rij[0], rij[1], rij[2]);
            const double aij = ai_prim + aj_prim;
            type1_rad_part(rad_all, LI+LJ, k, aij, ur);

            const double eij = exp(-ai_prim*r2ca - aj_prim*r2cb);
            const double eaij = eij * pow(-2.0*ai_prim, orderi) * pow(-2.0*aj_prim, orderj);
            const double ceij = eaij * ci[ip] * cj[jp];
            type1_rad_ang(rad_ang, LI+LJ, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }
    
    double fi[3*NFI_MAX];
    double fj[3*NFJ_MAX];
    cache_fac(fi, LI, rca);
    cache_fac(fj, LJ, rcb);

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int mi = ij%nfi;
        const int mj = ij/nfi;

        const int iy = _cart_pow_y[mi];
        const int iz = _cart_pow_z[mi];
        const int ix = LI - iy - iz;

        double* fx_i = fi + (ix+1)*ix/2;
        double* fy_i = fi + (iy+1)*iy/2 + nfi;
        double* fz_i = fi + (iz+1)*iz/2 + 2*nfi;

        const int jy = _cart_pow_y[mj];
        const int jz = _cart_pow_z[mj];
        const int jx = LJ - jy - jz;
        double* fx_j = fj + (jx+1)*jx/2;
        double* fy_j = fj + (jy+1)*jy/2 + nfj;
        double* fz_j = fj + (jz+1)*jz/2 + 2*nfj;

        // cache ifac and jfac in register
        double tmp = 0.0;
        for (int i1 = 0; i1 <= ix; i1++){
        for (int i2 = 0; i2 <= iy; i2++){
        for (int i3 = 0; i3 <= iz; i3++){
            double ifac = fx_i[i1] * fy_i[i2] * fz_i[i3];
            for (int jr = 0, j1 = 0; j1 <= jx; j1++){
            for (int j2 = 0; j2 <= jy; j2++){
            for (int j3 = 0; j3 <= jz; j3++, jr++){
                double jfac = fx_j[j1] * fy_j[j2] * fz_j[j3];
                const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
                tmp += ifac * jfac * rad_ang[ijr];
            }}}
        }}}
        gctr[ij] = tmp;
    }
    return;
}


__global__
void type1_cart_ip1(double *gctr, 
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
    
    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];
    for (int ij = threadIdx.x; ij < NF_MAX*NF_MAX*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();
    
    __shared__ double buf[NFI_MAX*NFJ_MAX];
    type1_cart_kernel<1,0>(buf, LI+1, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    _l_down(gctr_smem, buf, LI, LJ);
    if (LI > 0){
        type1_cart_kernel<0,0>(buf, LI-1, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        _l_up(gctr_smem, buf, LI, LJ);
    }

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gx + nao*nao;
        double *gz = gy + nao*nao;
        atomicAdd(gx + (i+ioff)*nao + (j+joff), gctr_smem[ij]);
        atomicAdd(gy + (i+ioff)*nao + (j+joff), gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz + (i+ioff)*nao + (j+joff), gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}

/*
__device__
double _contract(int ix, int iy, int iz, 
                double *fx_i, double *fy_i, double *fz_i,
                int jx, int jy, int jz, 
                double *fx_j, double *fy_j, double *fz_j,
                int LIJ1, double *rad_ang) {
    
    fy_j += (jy+1)*jy/2;
    fz_j += (jz+1)*jz/2;

    fx_i += (ix+1)*ix/2;
    fy_i += (iy+1)*iy/2;
    fz_i += (iz+1)*iz/2;
    
    double tmp = 0.0;
    for (int i1 = 0; i1 <= ix; i1++) {
    for (int i2 = 0; i2 <= iy; i2++) {
    for (int i3 = 0; i3 <= iz; i3++) {
        double ifac = fx_i[i1] * fy_i[i2] * fz_i[i3];
        for (int j1 = 0; j1 <= jx; j1++) {
        for (int j2 = 0; j2 <= jy; j2++) {
        for (int j3 = 0; j3 <= jz; j3++) {
            const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
            double jfac = fx_j[j1] * fy_j[j2] * fz_j[j3];
            tmp += ifac * jfac * rad_ang[ijr];
        }}}
    }}}
    return tmp;
}

__device__
void type1_contract(double *buf, int LI, int LJ, double *fi, double *fj, double *rad_ang) {
    const int LIJ1 = LI+LJ+1;
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x) {
        const int mi = ij%nfi;
        const int mj = ij/nfi;

        // Get angular momentum components from cartesian powers
        const int iy = _cart_pow_y[mi];
        const int iz = _cart_pow_z[mi];
        const int ix = LI - iy - iz;

        const int jy = _cart_pow_y[mj];
        const int jz = _cart_pow_z[mj];
        const int jx = LJ - jy - jz;
        
        double* fx_j = fj + (jx+1)*jx/2;
        double* fy_j = fj + (jy+1)*jy/2 + nfj;
        double* fz_j = fj + (jz+1)*jz/2 + 2*nfj;

        double* fx_i = fi + (ix+1)*ix/2;
        double* fy_i = fi + (iy+1)*iy/2 + nfi;
        double* fz_i = fi + (iz+1)*iz/2 + 2*nfi;
        
        double tmp = 0.0;
        for (int i1 = 0; i1 <= ix; i1++) {
        for (int i2 = 0; i2 <= iy; i2++) {
        for (int i3 = 0; i3 <= iz; i3++) {
            double ifac = fx_i[i1] * fy_i[i2] * fz_i[i3];
            for (int jr = 0, j1 = 0; j1 <= jx; j1++) {
            for (int j2 = 0; j2 <= jy; j2++) {
            for (int j3 = 0; j3 <= jz; j3++, jr++) {
                const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
                double jfac = fx_j[j1] * fy_j[j2] * fz_j[j3];
                tmp += ifac * jfac * rad_ang[ijr];
            }}}
        }}}

        buf[ij/THREADS] += tmp;
    }
}

__device__
void type1_cart_kernel(double *gctr, 
                        const int LI, const int LJ,
                        const int ish, const int jsh, const int ksh,
                        const int *ecpbas, const int *ecploc, 
                        const int *atm, const int *bas, const double *env)
{
    extern __shared__ double smem[];

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
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

    double fx_i[3*NF_MAX];
    double fx_j[3*NF_MAX];
    cache_fac(fx_i, LI, rca);
    cache_fac(fx_j, LJ, rcb);

    double *rad_ang = smem;

    double *rad_all = rad_ang + LIJ3;
    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];
    for (int ip = 0; ip < npi; ip++){
    for (int jp = 0; jp < npj; jp++){
        double rij[3];
        rij[0] = ai[ip] * rca[0] + aj[jp] * rcb[0];
        rij[1] = ai[ip] * rca[1] + aj[jp] * rcb[1];
        rij[2] = ai[ip] * rca[2] + aj[jp] * rcb[2];
        const double k = 2.0 * norm3d(rij[0], rij[1], rij[2]);
        const double aij = ai[ip] + aj[jp];

        type1_rad_part(rad_all, LI+LJ, k, aij, ur);

        const double eij = exp(-ai[ip]*r2ca - aj[jp]*r2cb);
        const double ceij = eij * ci[ip] * cj[jp];

        for (int i = threadIdx.x; i < LIJ3; i+=blockDim.x) {
            rad_ang[i] = 0;
        }
        __syncthreads();

        type1_rad_ang(rad_ang, LI+LJ, rij, rad_all, 1.0);
        __syncthreads();

        double reg_buf[NF_MAX*NF_MAX/THREADS+1] = {0.0};
        type1_contract(reg_buf, LI, LJ, fx_i, fx_j, rad_ang);

        // scale result with fac*ceij
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            gctr[ij/THREADS] += fac*ceij*reg_buf[ij/THREADS];
        }
    }}
}


__global__
void type1_general_cart(double *gctr, 
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

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;

    constexpr int nreg = (NF_MAX*NF_MAX+THREADS-1)/THREADS;
    double reg_gctr[nreg] = {0.0};
    type1_cart_kernel(reg_gctr, LI, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double tmp = reg_gctr[ij/THREADS];
        atomicAdd(gctr + i+ioff + (j+joff)*nao, tmp);
        if (ish != jsh){
            atomicAdd(gctr + (i+ioff)*nao + j+joff, tmp);
        }
    }
    return;
}

__global__
void type1_general_ip1_cart(double *gctr, 
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

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;

    constexpr int nreg = (NF_MAX*NF_MAX+THREADS-1)/THREADS;
    double reg_gctr[nreg] = {0.0};
    type1_cart_kernel(reg_gctr, LI, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double tmp = reg_gctr[ij/THREADS];
        atomicAdd(gctr + i+ioff + (j+joff)*nao, tmp);
    }
    return;
}
*/
