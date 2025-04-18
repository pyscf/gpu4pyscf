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

__device__
void type1_rad_part(double* __restrict__ rad_all, const int LIJ, double k, double aij, double ur)
{
    const double kaij = k / (2*aij);
    const double fac = kaij * kaij * aij;
    double r = 0.0;
    if (threadIdx.x < NGAUSS){
        r = r128[threadIdx.x];
    }
    double tmp = r - kaij;
    tmp = fac - aij*tmp*tmp;
    double bval[AO_LIJMAX+1];
    const int LIJ1 = LIJ + 1;
    double rur;
    if (ur == 0 || tmp > CUTOFF || tmp < -(EXPCUTOFF+6.+30.)) {
        rur = 0;
        for (int i = 0; i < LIJ1; i++){
            bval[i] = 0;
        }
    } else {
        rur = ur * exp(tmp);
        _ine(bval, LIJ, k*r);
    }

    for (int i = threadIdx.x; i < LIJ1*LIJ1; i+=blockDim.x){
        rad_all[i] = 0.0;
    }
    __syncthreads();
    for (int lab = 0; lab <= LIJ; lab++){
        if (lab > 0){
            rur *= r;
        }
        for (int i = lab%2; i <= LIJ; i+=2){
            block_reduce(rur*bval[i], rad_all+lab*LIJ1+i);
        }
    }
}
/*
template <int l> __device__
double type1_ang_nuc_l(const int i, const int j, const int k, double *unitr){
    double rxPow[l+1], ryPow[l+1], rzPow[l+1];
    rxPow[0] = ryPow[0] = rzPow[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rxPow[li] = rxPow[li - 1] * unitr[0];
        ryPow[li] = ryPow[li - 1] * unitr[1];
        rzPow[li] = rzPow[li - 1] * unitr[2];
    }

    double g[(l+1)*(l+2)/2];
    int index = 0;
    for (int li = l; li >= 0; li--) {
        for (int lj = l - li; lj >= 0; lj--) {
            int lk = l - li - lj;
            g[index++] = rxPow[li] * ryPow[lj] * rzPow[lk];
        }
    }

    double c[2*l+1];
    cart2sph<l>(c, g);
    double nuc[(l+1)*(l+2)/2];
    sph2cart<l>(nuc, c);

    double tmp = 0.0;
    for (int n = 0; n < (l+1)*(l+2)/2; n++){
        const int ps = _cart_pow_y[n];
        const int pt = _cart_pow_z[n];
        const int pr = l - ps - pt;
        if ((i+pr)%2 || (j+ps)%2 || (k+pt)%2){
            continue;
        }
        tmp += nuc[n] * int_unit_xyz(i+pr, j+ps, k+pt);
    }
    return tmp;
}
*/

__device__
void type1_rad_ang(double *rad_ang, const int LIJ, double *r, double *rad_all, const double fac)
{
    double unitr[3];
    if (r[0]*r[0] + r[1]*r[1] + r[2]*r[2] < 1e-16){
        unitr[0] = 0;
        unitr[1] = 0;
        unitr[2] = 0;
    } else {
        double norm_r = -rnorm3d(r[0], r[1], r[2]);
        unitr[0] = r[0] * norm_r;
        unitr[1] = r[1] * norm_r;
        unitr[2] = r[2] * norm_r;
    }

    // loop over i+j+k<=LIJ
    // TODO: find a closed form?
    for (int n = threadIdx.x; n < (LIJ+1)*(LIJ+1)*(LIJ+1); n+=blockDim.x){
        const int i = n/(LIJ+1)/(LIJ+1);
        const int j = n/(LIJ+1)%(LIJ+1);
        const int k = n%(LIJ+1);
        if (i+j+k > LIJ || (i+j+k)%2 == 1){
            continue;
        }
        // need_even to ensure (i+j+k+lmb) is even
        double s = 0.0;
        double *prad = rad_all + (i+j+k)*(LIJ+1);
        if (LIJ >= 0) s += prad[0] * type1_ang_nuc_l<0>(i, j, k, unitr);
        if (LIJ >= 2) s += prad[2] * type1_ang_nuc_l<2>(i, j, k, unitr);
        if (LIJ >= 4) s += prad[4] * type1_ang_nuc_l<4>(i, j, k, unitr);
        if (LIJ >= 6) s += prad[6] * type1_ang_nuc_l<6>(i, j, k, unitr);
        if (LIJ >= 8) s += prad[8] * type1_ang_nuc_l<8>(i, j, k, unitr);
        if (LIJ >= 10)s += prad[10]* type1_ang_nuc_l<10>(i, j, k, unitr);
        rad_ang[i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k] += fac*s;
        //atomicAdd(rad_ang + i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k, fac*s);
    }

    for (int n = threadIdx.x; n < (LIJ+1)*(LIJ+1)*(LIJ+1); n+=blockDim.x){
        const int i = n/(LIJ+1)/(LIJ+1);
        const int j = n/(LIJ+1)%(LIJ+1);
        const int k = n%(LIJ+1);
        if (i+j+k > LIJ || (i+j+k)%2 == 0){
            continue;
        }
        // need_even to ensure (i+j+k+lmb) is even
        double s = 0.0;
        double *prad = rad_all + (i+j+k)*(LIJ+1);
        if (LIJ >= 1) s += prad[1] * type1_ang_nuc_l<1>(i, j, k, unitr);
        if (LIJ >= 3) s += prad[3] * type1_ang_nuc_l<3>(i, j, k, unitr);
        if (LIJ >= 5) s += prad[5] * type1_ang_nuc_l<5>(i, j, k, unitr);
        if (LIJ >= 7) s += prad[7] * type1_ang_nuc_l<7>(i, j, k, unitr);
        if (LIJ >= 9) s += prad[9] * type1_ang_nuc_l<9>(i, j, k, unitr);
        rad_ang[i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k] += fac*s;
        //atomicAdd(rad_ang + i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k, fac*s);
    }
}

template <int LIJ> __device__
void type1_rad_ang(double *rad_ang, double *r, double *rad_all, const double fac)
{
    double unitr[3];
    if (r[0]*r[0] + r[1]*r[1] + r[2]*r[2] < 1e-16){
        unitr[0] = 0;
        unitr[1] = 0;
        unitr[2] = 0;
    } else {
        double norm_r = -rnorm3d(r[0], r[1], r[2]);
        unitr[0] = r[0] * norm_r;
        unitr[1] = r[1] * norm_r;
        unitr[2] = r[2] * norm_r;
    }

    // loop over i+j+k<=LIJ
    // TODO: find a closed form?
    for (int n = threadIdx.x; n < (LIJ+1)*(LIJ+1)*(LIJ+1); n+=blockDim.x){
        const int i = n/(LIJ+1)/(LIJ+1);
        const int j = n/(LIJ+1)%(LIJ+1);
        const int k = n%(LIJ+1);
        if (i+j+k > LIJ || (i+j+k)%2 == 1){
            continue;
        }
        // need_even to ensure (i+j+k+lmb) is even
        double s = 0.0;
        double *prad = rad_all + (i+j+k)*(LIJ+1);
        if constexpr (LIJ >= 0) s += prad[0] * type1_ang_nuc_l<0>(i, j, k, unitr);
        if constexpr (LIJ >= 2) s += prad[2] * type1_ang_nuc_l<2>(i, j, k, unitr);
        if constexpr (LIJ >= 4) s += prad[4] * type1_ang_nuc_l<4>(i, j, k, unitr);
        if constexpr (LIJ >= 6) s += prad[6] * type1_ang_nuc_l<6>(i, j, k, unitr);
        if constexpr (LIJ >= 8) s += prad[8] * type1_ang_nuc_l<8>(i, j, k, unitr);
        if constexpr (LIJ >= 10)s += prad[10]* type1_ang_nuc_l<10>(i, j, k, unitr);
        rad_ang[i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k] += fac*s;
        //atomicAdd(rad_ang + i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k, fac*s);
    }

    for (int n = threadIdx.x; n < (LIJ+1)*(LIJ+1)*(LIJ+1); n+=blockDim.x){
        const int i = n/(LIJ+1)/(LIJ+1);
        const int j = n/(LIJ+1)%(LIJ+1);
        const int k = n%(LIJ+1);
        if (i+j+k > LIJ || (i+j+k)%2 == 0){
            continue;
        }
        // need_even to ensure (i+j+k+lmb) is even
        double s = 0.0;
        double *prad = rad_all + (i+j+k)*(LIJ+1);
        if constexpr (LIJ >= 1) s += prad[1] * type1_ang_nuc_l<1>(i, j, k, unitr);
        if constexpr (LIJ >= 3) s += prad[3] * type1_ang_nuc_l<3>(i, j, k, unitr);
        if constexpr (LIJ >= 5) s += prad[5] * type1_ang_nuc_l<5>(i, j, k, unitr);
        if constexpr (LIJ >= 7) s += prad[7] * type1_ang_nuc_l<7>(i, j, k, unitr);
        if constexpr (LIJ >= 9) s += prad[9] * type1_ang_nuc_l<9>(i, j, k, unitr);
        rad_ang[i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k] += fac*s;
        //atomicAdd(rad_ang + i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k, fac*s);
    }
}

template <int LI, int LJ> __global__
void type1_cart(double *gctr,
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
    __shared__ double rad_ang[LIJ1*LIJ1*LIJ1];
    set_shared_memory(rad_ang, LIJ1*LIJ1*LIJ1);

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];
    for (int ip = 0; ip < npi; ip++){
        for (int jp = 0; jp < npj; jp++){
            double rij[3];
            rij[0] = ai[ip] * rca[0] + aj[jp] * rcb[0];
            rij[1] = ai[ip] * rca[1] + aj[jp] * rcb[1];
            rij[2] = ai[ip] * rca[2] + aj[jp] * rcb[2];
            const double k = 2.0 * norm3d(rij[0], rij[1], rij[2]);
            const double aij = ai[ip] + aj[jp];

            __shared__ double rad_all[LIJ1*LIJ1];
            type1_rad_part(rad_all, LI+LJ, k, aij, ur);
            __syncthreads();

            const double eij = exp(-ai[ip]*r2ca - aj[jp]*r2cb);
            const double ceij = eij * ci[ip] * cj[jp];
            type1_rad_ang<LI+LJ>(rad_ang, rij, rad_all, fac*ceij);
            //type1_rad_ang(rad_ang, LI+LJ, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    double fi[3*nfi];
    cache_fac<LI>(fi, rca);
    double fj[3*nfj];
    cache_fac<LJ>(fj, rcb);

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int mi = ij%nfi;
        const int mj = ij/nfi;

        // TODO: Read the same constant memory in each warp
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

        double tmp = 0.0;
        for (int i1 = 0; i1 <= ix; i1++){
        for (int i2 = 0; i2 <= iy; i2++){
        for (int i3 = 0; i3 <= iz; i3++){
            const double ifac = fx_i[i1] * fy_i[i2] * fz_i[i3];
            for (int j1 = 0; j1 <= jx; j1++){
            for (int j2 = 0; j2 <= jy; j2++){
            for (int j3 = 0; j3 <= jz; j3++){
                const double jfac = fx_j[j1] * fy_j[j2] * fz_j[j3];
                const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
                tmp += ifac * jfac * rad_ang[ijr];
            }}}
        }}}

        const int ioff = ao_loc[ish];
        const int joff = ao_loc[jsh];
        atomicAdd(gctr + mi+ioff + (mj+joff)*nao, tmp);
        if (ish != jsh){
            atomicAdd(gctr + (mi+ioff)*nao + mj+joff, tmp);
        }
    }
    return;
}


__global__
void type1_cart(double *gctr,
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

    double *rad_ang = smem;
    set_shared_memory(rad_ang, (LI+LJ+1)*(LI+LJ+1)*(LI+LJ+1));

    double *rad_all = rad_ang + (LI+LJ+1)*(LI+LJ+1)*(LI+LJ+1);
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
            __syncthreads();

            const double eij = exp(-ai_prim*r2ca - aj_prim*r2cb);
            const double ceij = eij * ci[ip] * cj[jp];
            type1_rad_ang(rad_ang, LI+LJ, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }

    constexpr int nreg = (NF_MAX*NF_MAX+THREADS-1)/THREADS;
    double reg_gctr[nreg];
    for (int i = 0; i < nreg; i++){
        reg_gctr[i] = 0.0;
    }
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        double fi[3*NF_MAX];
        cache_fac(fi, LI, rca);
        double fj[3*NF_MAX];
        cache_fac(fj, LJ, rcb);

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
            const double ifac = fx_i[i1] * fy_i[i2] * fz_i[i3];
            for (int jr = 0, j1 = 0; j1 <= jx; j1++){
            for (int j2 = 0; j2 <= jy; j2++){
            for (int j3 = 0; j3 <= jz; j3++, jr++){
                const int LIJ1 = LI+LJ+1;
                const double jfac = fx_j[j1] * fy_j[j2] * fz_j[j3];
                const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
                tmp += ifac * jfac * rad_ang[ijr];
            }}}
        }}}
        reg_gctr[ij/THREADS] += tmp;
    }

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

