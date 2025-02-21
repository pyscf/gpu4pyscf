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

template <int LIC> __device__
void type2_facs_rad(double* facs, const int np, double rca,
                    const double *ci, const double *ai){
    double r = r128[threadIdx.x] - rca;
    double r2 = r*r;
    for (int j = 0; j <= LIC; j++){
        facs[j] = 0.0;
    }

    for (int ip = 0; ip < np; ip++){
        double ka = 2.0 * ai[ip] * rca;
        double ar2 = ai[ip] * r2;
        double buf[LIC+1];
        if (ar2 > EXPCUTOFF + 6.0){
            for (int j = 0; j <= LIC; j++){
                buf[j] = 0.0;
            }
        } else {
            double t1 = exp(-ar2);
            _ine<LIC>(buf, ka*r128[threadIdx.x]);
            for (int j = 0; j <= LIC; j++){
                buf[j] *= t1;
            }
        }
        for (int j = 0; j <= LIC; j++){
            facs[j] += ci[ip] * buf[j];
        }
    }
}

template <int LI, int LC> __device__
void type2_facs_omega(double* omega, double *r){
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
    constexpr int LIC1 = LI+LC+1;
    constexpr int LCC1 = LC*2+1;

    double omega_nuc[CART_CUM];

    ang_nuc_part<LI+LC>(omega_nuc, unitr[0], unitr[1], unitr[2]);
    // The total number of (i,j,k) is tetrahedral number
    // store (i,j,k) such that i+j+k<=LI
    // store (i+j+k+lc)%2 = 0 only
    // up to 12600 Bytes
    constexpr int BLK = (LIC1+1)/2 * LCC1;
    //for (int i = threadIdx.x; i <= LI; i+=blockDim.x){
    //for (int j = 0; j <= LI-i; j++){
    //for (int k = 0; k <= LI-i-j; k++){
    constexpr int LI1 = LI + 1;
    for (int n = threadIdx.x; n < LI1*LI1*LI1; n+=blockDim.x){
        const int i = n/LI1/LI1;
        const int j = n/LI1%LI1;
        const int k = n%LI1;
        if (i+j+k > LI){
            continue;
        }
        const int need_even = (LC+i+j+k)%2;
        const int ioff = (LI-i)*(LI-i+1)*(LI-i+2)/6;
        const int joff = (LI-i-j)*(LI-i-j+1)/2;
        double *pomega = omega + (ioff+joff+k)*BLK;
        for (int lmb = need_even; lmb <= LI+LC; lmb+=2){
            double *pnuc = omega_nuc + _offset_cart[lmb];
            double buf[(LC+1)*(LC+2)/2];
            for (int m = 0; m < (LC+1)*(LC+2)/2; m++) buf[m] = 0.0;
            for (int n = 0; n < (lmb+1)*(lmb+2)/2; n++){
                const int ps = _cart_pow_y[n];
                const int pt = _cart_pow_z[n];
                const int pr = lmb - ps - pt;
                double nuc = pnuc[n];
                for (int m = 0; m < (LC+1)*(LC+2)/2; m++){
                    const int pv = _cart_pow_y[m];
                    const int pw = _cart_pow_z[m];
                    const int pu = LC - pv - pw;
                    buf[m] += nuc * int_unit_xyz(i+pu+pr, j+pv+ps, k+pw+pt);
                }
            }
            for (int m = 0; m < (LC+1)*(LC+2)/2; m++) buf[m] *= 4.0 * M_PI;
            cart2sph<LC>(pomega, buf);
            pomega += LCC1;
        }
    }
    __syncthreads();
}

template <int LI, int LC> __device__
void type2_ang(double *facs, double *fx, double *omega, int m){
    constexpr int LI1 = LI+1;
    constexpr int nfi = LI1*(LI1+1)/2;
    constexpr int LCC1 = (2*LC+1);
    constexpr int LIC1 = LI+LC+1;
    constexpr int BLK = (LIC1+1)/2 * LCC1;

    double *fy = fx + nfi;
    double *fz = fy + nfi;

    for (int i = threadIdx.x; i < LI1*nfi*LIC1; i+=blockDim.x){
        facs[i] = 0.0;
    }
    __syncthreads();

    // i,j,k,ijkmn->(i+j+k)pmn
    for (int p = threadIdx.x; p < (LI+1)*(LI+2)/2; p+=blockDim.x){
        const int iy = _cart_pow_y[p];
        const int iz = _cart_pow_z[p];
        const int ix = LI - iy - iz;
        for (int i = 0; i <= ix; i++){
        for (int j = 0; j <= iy; j++){
        for (int k = 0; k <= iz; k++){
            const int need_even = (LC+i+j+k)%2;
            const int xoffset = (ix+1)*ix/2;
            const int yoffset = (iy+1)*iy/2;
            const int zoffset = (iz+1)*iz/2;
            double fac = fx[xoffset+i] * fy[yoffset+j] * fz[zoffset+k];

            const int ioff = (LI-i)*(LI-i+1)*(LI-i+2)/6;
            const int joff = (LI-i-j)*(LI-i-j+1)/2;
            double *pomega = omega + (ioff+joff+k)*BLK;
            double *pfacs = facs + ((i+j+k)*nfi+p)*LIC1;
            for (int n = need_even; n < LIC1; n+=2){
                atomicAdd(pfacs+n, fac * pomega[n/2*LCC1+m]);
            }
        }}}
    }
    __syncthreads();
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
    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
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

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    double radi[LIC1];
    double radj[LJC1];
    type2_facs_rad<LI+LC>(radi, npi, dca, ci, ai);
    type2_facs_rad<LJ+LC>(radj, npj, dcb, cj, aj);

    __shared__ double rad_all[(LI+LJ+1) * LIC1 * LJC1]; // up to 5832 Bytes
    for (int i = threadIdx.x; i < (LI+LJ+1)*LIC1*LJC1; i+=blockDim.x){
        rad_all[i] = 0.0;
    }
    __syncthreads();

    int ir = threadIdx.x;
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            // TODO: block reduction
            block_reduce(radi[i]*radj[j]*ur, prad+i*LJC1+j);
            //atomicAdd(prad+i*LJC1+j, radi[i] * radj[j] * ur);
        }}
        ur *= r128[ir];
    }
    __syncthreads();

    constexpr int BLKI = (LIC1+1)/2 * LCC1;
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    __shared__ double omegai[LI1*(LI1+1)*(LI1+2)/6 * BLKI]; // up to 12600 Bytes
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];

    type2_facs_omega<LI, LC>(omegai, rca);
    type2_facs_omega<LJ, LC>(omegaj, rcb);

    double fi[nfi*3];
    double fj[nfj*3];
    cache_fac<LI>(fi, rca);
    cache_fac<LJ>(fj, rcb);

    __shared__ double angi[LI1*nfi*LIC1]; // up to 5400 Bytes, further compression
    __shared__ double angj[LJ1*nfj*LJC1];

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang<LI, LC>(angi, fi, omegai, m);
        type2_ang<LJ, LC>(angj, fj, omegaj, m);

        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                double *prad = rad_all + (k+l)*LIC1*LJC1;
                // TODO: cache angi and angj in registers
                for (int p = 0; p < LIC1; p++){
                for (int q = 0; q < LJC1; q++){
                    s += prad[p*LJC1+q] * pangi[p] * pangj[q];
                }}
            }}
            const double tmp = fac*s;
            atomicAdd(gctr + i+ioff + (j+joff)*nao, tmp);
            if (ish != jsh){
                atomicAdd(gctr + (i+ioff)*nao + j+joff, tmp);
            }
        }
        __syncthreads();
    }
    return;
}
