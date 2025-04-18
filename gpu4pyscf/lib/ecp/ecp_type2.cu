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

template <int order> __device__
void type2_facs_rad(double* facs, const int LIC, const int np, const double rca,
                    const double *ci, const double *ai){
    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    const double r = root - rca;
    const double r2 = r*r;
    for (int j = 0; j <= LIC; j++){
        facs[j] = 0.0;
    }

    for (int ip = 0; ip < np; ip++){
        const double ka = 2.0 * ai[ip] * rca;
        const double ar2 = ai[ip] * r2;

        double buf[AO_LMAX+ECP_LMAX+order+1];
        if (ar2 > EXPCUTOFF + 6.0){
            for (int j = 0; j <= LIC; j++){
                buf[j] = 0.0;
            }
        } else {
            const double t1 = exp(-ar2);
            _ine(buf, LIC, ka*root);
            for (int j = 0; j <= LIC; j++){
                buf[j] *= t1;
            }
        }
        const double c = pow(-2.0*ai[ip], order) * ci[ip];
        for (int j = 0; j <= LIC; j++){
            facs[j] += c * buf[j];
        }
    }
}

__device__
void type2_facs_omega(double* __restrict__ omega, const int LI, const int LC, double *r){
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

    // LC + (i+j+k) + (LI + LC) needs to be even
    // When i+j+k + LC is even
    for (int n = threadIdx.x; n < (LI+1)*(LI+1)*(LI+1); n+=blockDim.x){
        const int i = n/(LI+1)/(LI+1);
        const int j = n/(LI+1)%(LI+1);
        const int k = n%(LI+1);
        if (i+j+k > LI || (i+j+k+LC)%2 == 1){
            continue;
        }

        const int LI_i = LI-i;
        const int ioff = (LI_i)*(LI_i+1)*(LI_i+2)/6;
        const int joff = (LI_i-j)*(LI_i-j+1)/2;
        const int blk = (LI+LC+2)/2 * (LC*2+1);
        double *pomega = omega + (ioff+joff+k)*blk;

        //for (int lmb = need_even; lmb <= LI+LC; lmb+=2){
        if (LI+LC >= 0)  {type2_ang_nuc_l<0>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 2)  {type2_ang_nuc_l<2>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 4)  {type2_ang_nuc_l<4>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 6)  {type2_ang_nuc_l<6>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 8)  {type2_ang_nuc_l<8>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 10) {type2_ang_nuc_l<10>(pomega, LC, i, j, k, unitr); pomega+=(2*LC+1);}
    }

    // When i+j+k + LC is odd
    for (int n = threadIdx.x; n < (LI+1)*(LI+1)*(LI+1); n+=blockDim.x){
        const int i = n/(LI+1)/(LI+1);
        const int j = n/(LI+1)%(LI+1);
        const int k = n%(LI+1);
        if (i+j+k > LI || (i+j+k+LC)%2 == 0){
            continue;
        }
        const int LI_i = LI-i;
        const int ioff = (LI_i)*(LI_i+1)*(LI_i+2)/6;
        const int joff = (LI_i-j)*(LI_i-j+1)/2;
        const int blk = (LI+LC+2)/2 * (LC*2+1);
        double *pomega = omega + (ioff+joff+k)*blk;

        //for (int lmb = need_even; lmb <= LI+LC; lmb+=2){
        if (LI+LC >= 1)  {type2_ang_nuc_l<1>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 3)  {type2_ang_nuc_l<3>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 5)  {type2_ang_nuc_l<5>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 7)  {type2_ang_nuc_l<7>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if (LI+LC >= 9)  {type2_ang_nuc_l<9>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
    }
    __syncthreads();
}

template <int LI, int LC> __device__
void type2_facs_omega(double* __restrict__ omega, double *r){
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

    // LC + (i+j+k) + (LI + LC) needs to be even
    // When i+j+k + LC is even
    for (int n = threadIdx.x; n < (LI+1)*(LI+1)*(LI+1); n+=blockDim.x){
        const int i = n/(LI+1)/(LI+1);
        const int j = n/(LI+1)%(LI+1);
        const int k = n%(LI+1);
        if (i+j+k > LI || (i+j+k+LC)%2 == 1){
            continue;
        }

        const int LI_i = LI-i;
        const int ioff = (LI_i)*(LI_i+1)*(LI_i+2)/6;
        const int joff = (LI_i-j)*(LI_i-j+1)/2;
        constexpr int blk = (LI+LC+2)/2 * (LC*2+1);
        double *pomega = omega + (ioff+joff+k)*blk;

        //for (int lmb = need_even; lmb <= LI+LC; lmb+=2){
        if constexpr (LI+LC >= 0)  {type2_ang_nuc_l<0>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 2)  {type2_ang_nuc_l<2>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 4)  {type2_ang_nuc_l<4>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 6)  {type2_ang_nuc_l<6>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 8)  {type2_ang_nuc_l<8>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 10) {type2_ang_nuc_l<10>(pomega, LC, i, j, k, unitr); pomega+=(2*LC+1);}
    }

    // When i+j+k + LC is odd
    for (int n = threadIdx.x; n < (LI+1)*(LI+1)*(LI+1); n+=blockDim.x){
        const int i = n/(LI+1)/(LI+1);
        const int j = n/(LI+1)%(LI+1);
        const int k = n%(LI+1);
        if (i+j+k > LI || (i+j+k+LC)%2 == 0){
            continue;
        }
        const int LI_i = LI-i;
        const int ioff = (LI_i)*(LI_i+1)*(LI_i+2)/6;
        const int joff = (LI_i-j)*(LI_i-j+1)/2;
        constexpr int blk = (LI+LC+2)/2 * (LC*2+1);
        double *pomega = omega + (ioff+joff+k)*blk;

        //for (int lmb = need_even; lmb <= LI+LC; lmb+=2){
        if constexpr (LI+LC >= 1)  {type2_ang_nuc_l<1>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 3)  {type2_ang_nuc_l<3>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 5)  {type2_ang_nuc_l<5>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 7)  {type2_ang_nuc_l<7>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr (LI+LC >= 9)  {type2_ang_nuc_l<9>(pomega, LC, i, j, k, unitr);  pomega+=(2*LC+1);}
    }
}


__device__
void type2_ang(double * __restrict__ facs, const int LI, const int LC, double *rca, double *omega){
    const int LI1 = LI+1;
    const int nfi = LI1*(LI1+1)/2;
    const int LCC1 = (2*LC+1);
    const int LIC1 = LI+LC+1;
    const int BLK = (LIC1+1)/2 * LCC1;

    constexpr int NF_MAX_IP = (AO_LMAX_IP+1)*(AO_LMAX_IP+2)/2;
    double fi[NF_MAX_IP*3];
    cache_fac(fi, LI, rca);

    // i,j,k,ijkmn->(i+j+k)pmn
    for (int pmn = threadIdx.x; pmn < nfi*LIC1; pmn+=blockDim.x){
        const int m = pmn/nfi;
        const int p = pmn%nfi;

        const int iy = _cart_pow_y[p];
        const int iz = _cart_pow_z[p];
        const int ix = LI - iy - iz;

        const int ix_off = (ix+1)*ix/2;
        const int iy_off = (iy+1)*iy/2 + nfi;
        const int iz_off = (iz+1)*iz/2 + nfi*2;

        double ang_pmn[AO_LMAX_IP+1];
        for (int i = 0; i < AO_LMAX_IP+1; i++){
            ang_pmn[i] = 0.0;
        }

        for (int i = 0; i <= ix; i++){
        for (int j = 0; j <= iy; j++){
        for (int k = 0; k <= iz; k++){
            const int ijk = i+j+k;
            const double fac = fi[i+ix_off] * fi[j+iy_off] * fi[k+iz_off];
            const int LI_i = LI-i;
            const int ioff = (LI_i)*(LI_i+1)*(LI_i+2)/6;
            const int joff = (LI_i-j)*(LI_i-j+1)/2;
            double *pomega = omega + (ioff+joff+k)*BLK;
            if ((LC+ijk)%2 == m%2){
                ang_pmn[ijk] += fac * pomega[m/2*LCC1];
            }
        }}}

        for (int i = 0; i <= LI; i++){
            facs[i*nfi*LIC1 + p*LIC1 + m] = ang_pmn[i];
        }
    }
}

template <int LI, int LC> __device__
void type2_ang(double * __restrict__ facs, double *rca, double *omega){
    constexpr int LI1 = LI+1;
    constexpr int nfi = LI1*(LI1+1)/2;
    constexpr int LCC1 = (2*LC+1);
    constexpr int LIC1 = LI+LC+1;
    constexpr int BLK = (LIC1+1)/2 * LCC1;

    constexpr int NF = (LI+1)*(LI+2)/2;
    double fi[NF*3];
    cache_fac<LI>(fi, rca);

    // i,j,k,ijkmn->(i+j+k)pmn
    for (int pmn = threadIdx.x; pmn < nfi*LIC1; pmn+=blockDim.x){
        const int m = pmn/nfi;
        const int p = pmn%nfi;

        const int iy = _cart_pow_y[p];
        const int iz = _cart_pow_z[p];
        const int ix = LI - iy - iz;

        const int ix_off = (ix+1)*ix/2;
        const int iy_off = (iy+1)*iy/2 + nfi;
        const int iz_off = (iz+1)*iz/2 + nfi*2;

        double ang_pmn[LI+1];
        for (int i = 0; i < LI+1; i++){
            ang_pmn[i] = 0.0;
        }

        for (int i = 0; i <= ix; i++){
        for (int j = 0; j <= iy; j++){
        for (int k = 0; k <= iz; k++){
            const int ijk = i+j+k;
            const double fac = fi[i+ix_off] * fi[j+iy_off] * fi[k+iz_off];
            const int LI_i = LI-i;
            const int ioff = (LI_i)*(LI_i+1)*(LI_i+2)/6;
            const int joff = (LI_i-j)*(LI_i-j+1)/2;
            double *pomega = omega + (ioff+joff+k)*BLK;
            if ((LC+ijk)%2 == m%2){
                ang_pmn[ijk] += fac * pomega[m/2*LCC1];
            }
        }}}

        for (int i = 0; i <= LI; i++){
            facs[i*nfi*LIC1 + p*LIC1 + m] = ang_pmn[i];
        }
    }
}

template <int LI, int LJ, int LC> __global__
void type2_cart(double * __restrict__ gctr,
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

    constexpr int LI1 = LI+1;
    constexpr int LJ1 = LJ+1;
    constexpr int LIC1 = LI+LC+1;
    constexpr int LJC1 = LJ+LC+1;
    constexpr int LCC1 = (2*LC+1);

    constexpr int BLKI = (LIC1+1)/2 * LCC1;
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    __shared__ double omegai[LI1*(LI1+1)*(LI1+2)/6 * BLKI];
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];

    type2_facs_omega<LI, LC>(omegai, rca);
    type2_facs_omega<LJ, LC>(omegaj, rcb);
    __syncthreads();

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    const double dca = norm3d(rca[0], rca[1], rca[2]);
    double radi[LIC1];
    type2_facs_rad<0>(radi, LI+LC, npi, dca, ci, ai);

    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    double radj[LJC1];
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    __shared__ double rad_all[(LI+LJ+1) * LIC1 * LJC1];
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur, prad+i*LJC1+j);
        }}
        ur *= root;
    }
    __syncthreads();

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;

    __shared__ double angi[LI1*nfi*LIC1];
    __shared__ double angj[LJ1*nfj*LJC1];

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    constexpr int nreg = (nfi*nfj + THREADS - 1)/THREADS;
    double reg_gctr[nreg];
    for (int i = 0; i < nreg; i++){
        reg_gctr[i] = 0.0;
    }

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang<LI, LC>(angi, rca, omegai+m);
        type2_ang<LJ, LC>(angj, rcb, omegaj+m);
        __syncthreads();
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                double *prad = rad_all + (k+l)*LIC1*LJC1;
                double reg_angi[LIC1];
                double reg_angj[LJC1];
                for (int p = 0; p < LIC1; p++){reg_angi[p] = pangi[p];}
                for (int q = 0; q < LJC1; q++){reg_angj[q] = pangj[q];}
                for (int p = 0; p < LIC1; p++){
                for (int q = 0; q < LJC1; q++){
                    s += prad[p*LJC1+q] * reg_angi[p] * reg_angj[q];
                }}
            }}
            reg_gctr[ij/THREADS] += fac*s;
        }
        __syncthreads();
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

__global__
void type2_cart(double * __restrict__ gctr,
                const int LI, const int LJ, const int LC,
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

    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];

    double* omegai = smem + (LI+LJ+1) * (LI+LC+1) * (LJ+LC+1);
    double* omegaj = omegai + (LI+LC+2)/2 * (LI+1)*(LI+2)*(LI+3)/6 * (2*LC+1);

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    double radi[AO_LMAX+ECP_LMAX+1];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    type2_facs_rad<0>(radi, LI+LC, npi, dca, ci, ai);

    double radj[AO_LMAX+ECP_LMAX+1];
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    double* rad_all = smem;
    set_shared_memory(rad_all, (LI+LJ+1)*(LI+LC+1)*(LJ+LC+1));
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*(LI+LC+1)*(LJ+LC+1);
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur, prad+i*(LJ+LC+1)+j);
        }}
        ur *= root;
    }

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    double* angi = omegaj + (LJ+LC+2)/2 * (LJ+1)*(LJ+2)*(LJ+3)/6 * (2*LC+1);
    double* angj = angi + (LI+1)*nfi*(LI+LC+1);

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    constexpr int nreg = (NF_MAX*NF_MAX + THREADS - 1)/THREADS;
    double reg_gctr[nreg];
    for (int i = 0; i < nreg; i++){
        reg_gctr[i] = 0.0;
    }

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < 2*LC+1; m++){
        type2_ang(angi, LI, LC, rca, omegai+m);
        type2_ang(angj, LJ, LC, rcb, omegaj+m);
        __syncthreads();

        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                const int LIC1 = LI+LC+1;
                const int LJC1 = LJ+LC+1;
                double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                double *prad = rad_all + (k+l)*LIC1*LJC1;
                double reg_angi[AO_LMAX+ECP_LMAX+1];
                double reg_angj[AO_LMAX+ECP_LMAX+1];
                for (int p = 0; p < LIC1; p++){reg_angi[p] = pangi[p];}
                for (int q = 0; q < LJC1; q++){reg_angj[q] = pangj[q];}
                for (int p = 0; p < LIC1; p++){
                for (int q = 0; q < LJC1; q++){
                    s += prad[p*LJC1+q] * reg_angi[p] * reg_angj[q];
                }}
            }}
            reg_gctr[ij/THREADS] += fac*s;
        }
        __syncthreads();
    }

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    double *gctr_ij = gctr + ioff + joff*nao;
    double *gctr_ji = gctr + joff + ioff*nao;
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double tmp = reg_gctr[ij/THREADS];
        atomicAdd(gctr_ij + i + j*nao, tmp);
        if (ish != jsh){
            atomicAdd(gctr_ji + i*nao + j, tmp);
        }
    }

    return;
}
