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
void type2_facs_rad(double* facs, const int LIC, const int np, double rca,
                    const double *ci, const double *ai){
    const double r = r128[threadIdx.x] - rca;
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
            _ine(buf, LIC, ka*r128[threadIdx.x]);
            for (int j = 0; j <= LIC; j++){
                buf[j] *= t1;
            }
        }
        if (threadIdx.x == 0){
            printf("%f %f %f\n", r128[threadIdx.x], buf[0], facs[0]);
        }
        const double c = pow(-2.0*ai[ip], order) * ci[ip];
        for (int j = 0; j <= LIC; j++){
            facs[j] += c * buf[j];
        }
        if (threadIdx.x == 0){
            printf("%f %f %f %f\n", r128[threadIdx.x], buf[0], c , facs[0]);
        }
    }
}

template <int l> __device__ 
void type2_ang_nuc_l(double *omega, const int lc, 
                    const int i, const int j, const int k, double *unitr){
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
    cart2sph(c, l, g);
    double nuc_cart[(l+1)*(l+2)/2];
    sph2cart(nuc_cart, l, c);
    
    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    for (int n = 0; n < (l+1)*(l+2)/2; n++){
        const int ps = _cart_pow_y[n];
        const int pt = _cart_pow_z[n];
        const int pr = l - ps - pt;
        const double nuc = nuc_cart[n];
        for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
            const int pv = _cart_pow_y[m];
            const int pw = _cart_pow_z[m];
            const int pu = lc - pv - pw;
            buf[m] += nuc * int_unit_xyz(i+pu+pr, j+pv+ps, k+pw+pt);
        }
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}

__device__ 
void type2_ang_nuc(double *omega, const int lmb, const int lc,
                    const int i, const int j, const int k, double *unitr){
    switch(lmb){
        case 0: {type2_ang_nuc_l<0>(omega, lc, i, j, k, unitr); break;}
        case 1: {type2_ang_nuc_l<1>(omega, lc, i, j, k, unitr); break;}
        case 2: {type2_ang_nuc_l<2>(omega, lc, i, j, k, unitr); break;}
        case 3: {type2_ang_nuc_l<3>(omega, lc, i, j, k, unitr); break;}
        case 4: {type2_ang_nuc_l<4>(omega, lc, i, j, k, unitr); break;}
        case 5: {type2_ang_nuc_l<5>(omega, lc, i, j, k, unitr); break;}
        case 6: {type2_ang_nuc_l<6>(omega, lc, i, j, k, unitr); break;}
        case 7: {type2_ang_nuc_l<7>(omega, lc, i, j, k, unitr); break;}
        case 8: {type2_ang_nuc_l<8>(omega, lc, i, j, k, unitr); break;}
        case 9: {type2_ang_nuc_l<9>(omega, lc, i, j, k, unitr); break;}
        case 10: {type2_ang_nuc_l<10>(omega, lc, i, j, k, unitr); break;}
        default: {printf("l = %d is not supported\n", lmb);}
    }
}

__device__
void type2_facs_omega(double* omega, const int LI, const int LC, double *r){
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
    const int LIC1 = LI+LC+1;
    const int LCC1 = LC*2+1;

    const int BLK = (LIC1+1)/2 * LCC1;
    const int LI1 = LI + 1;
    for (int nijk = threadIdx.x; nijk < LI1*LI1*LI1; nijk+=blockDim.x){
        const int i = nijk/LI1/LI1;
        const int j = nijk/LI1%LI1;
        const int k = nijk%LI1;
        const int ijk = i+j+k;
        if (ijk > LI){
            continue;
        }
        const int need_even = (LC+ijk)%2;
        const int LI_i = LI-i;
        const int ioff = (LI_i)*(LI_i+1)*(LI_i+2)/6;
        const int joff = (LI_i-j)*(LI_i-j+1)/2;
        double *pomega = omega + (ioff+joff+k)*BLK;

        for (int lmb = need_even; lmb <= LI+LC; lmb+=2){
            type2_ang_nuc(pomega, lmb, LC, i, j, k, unitr);
            pomega += LCC1;
        }
    }
}

__device__
void type2_ang(double *facs, const int LI, const int LC, double *fi, double *omega){
    const int LI1 = LI+1;
    const int nfi = LI1*(LI1+1)/2;
    const int LCC1 = (2*LC+1);
    const int LIC1 = LI+LC+1;
    const int BLK = (LIC1+1)/2 * LCC1;

    // i,j,k,ijkmn->(i+j+k)pmn
    for (int pmn = threadIdx.x; pmn < nfi*LIC1; pmn+=blockDim.x){
        const int m = pmn/nfi;
        const int p = pmn%nfi;
    
        const int iy = _cart_pow_y[p];
        const int iz = _cart_pow_z[p];
        const int ix = LI - iy - iz;
        
        double *fx = fi + (ix+1)*ix/2;
        double *fy = fi + (iy+1)*iy/2 + nfi;
        double *fz = fi + (iz+1)*iz/2 + nfi*2;
        
        double ang_pmn[AO_LMAX_IP+1] = {0.0};
        for (int i = 0; i <= ix; i++){
        for (int j = 0; j <= iy; j++){
        for (int k = 0; k <= iz; k++){
            const int ijk = i+j+k;
            const double fac = fx[i] * fy[j] * fz[k];

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
void type2_cart(double *gctr, 
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
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    constexpr int LI1 = LI+1;
    constexpr int LJ1 = LJ+1;
    constexpr int LIC1 = LI+LC+1;
    constexpr int LJC1 = LJ+LC+1;
    constexpr int LCC1 = (2*LC+1);
    
    double radi[LIC1];
    double radj[LJC1];
    type2_facs_rad<0>(radi, LI+LC, npi, dca, ci, ai);
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    __shared__ double rad_all[(LI+LJ+1) * LIC1 * LJC1];
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }
    double ur_tmp = ur;
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur_tmp, prad+i*LJC1+j);
        }}
        int ir = threadIdx.x;
        ur_tmp *= r128[ir];
    }
    __syncthreads();
    if (threadIdx.x == 0){
        printf("%f %f %f\n", rad_all[0], radi[0], radj[0]);
    }
    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    double fi[nfi*3];
    double fj[nfj*3];
    cache_fac(fi, LI, rca);
    cache_fac(fj, LJ, rcb);
    
    constexpr int BLKI = (LIC1+1)/2 * LCC1;
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    __shared__ double omegai[LI1*(LI1+1)*(LI1+2)/6 * BLKI]; // up to 12600 Bytes
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();

    __shared__ double angi[LI1*nfi*LIC1]; // up to 5400 Bytes, further compression
    __shared__ double angj[LJ1*nfj*LJC1];

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    double reg_gctr[nfi*nfj/THREADS+1] = {0.0};

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang(angi, LI, LC, fi, omegai+m);
        type2_ang(angj, LJ, LC, fj, omegaj+m);
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
void type2_cart(double *gctr, 
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
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    const int LI1 = LI+1;
    const int LJ1 = LJ+1;
    const int LIC1 = LI+LC+1;
    const int LJC1 = LJ+LC+1;
    const int LCC1 = (2*LC+1);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    double radi[AO_LMAX+ECP_LMAX+1] = {0.0};
    double radj[AO_LMAX+ECP_LMAX+1] = {0.0};
    type2_facs_rad<0>(radi, LI+LC, npi, dca, ci, ai);
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);
    if (threadIdx.x == 0){
        printf("%f %f %f %f %f %f %f %f\n", dca, dcb, radi[0], radj[0], ci[0], cj[0], ai[0], aj[0]);
    }

    double* rad_all = smem;
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);

    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur, prad+i*LJC1+j);
        }}
        int ir = threadIdx.x;
        ur *= r128[ir];
    }
    __syncthreads();

    const int BLKI = (LIC1+1)/2 * LCC1;
    const int BLKJ = (LJC1+1)/2 * LCC1;

    double* omegai = smem + (LI+LJ+1) * LIC1 * LJC1;
    double* omegaj = omegai + LI1*(LI1+1)*(LI1+2)/6 * BLKI;

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();
    
    double fi[NF_MAX*3];
    double fj[NF_MAX*3];
    cache_fac(fi, LI, rca);
    cache_fac(fj, LJ, rcb);

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    double* angi = omegaj + LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ;
    double* angj = angi + LI1*nfi*LIC1;

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    constexpr int nreg = (NF_MAX*NF_MAX + THREADS - 1)/THREADS;
    double reg_gctr[nreg] = {0.0};
    
    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang(angi, LI, LC, fi, omegai+m);
        type2_ang(angj, LJ, LC, fj, omegaj+m);
        __syncthreads();
        return;
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
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
