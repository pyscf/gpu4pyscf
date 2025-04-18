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


template <int orderi, int orderj, int LI, int LJ, int LC> __device__
void type2_cart_unrolled_kernel(double *gctr,
                const int ish, const int jsh, const int ksh,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
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

    __shared__ double rad_all[(LI+LJ+1)*LIC1*LJC1];
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    double radi[LIC1];
    type2_facs_rad<orderi>(radi, LI+LC, npi, dca, ci, ai);

    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    double radj[LJC1];
    type2_facs_rad<orderj>(radj, LJ+LC, npj, dcb, cj, aj);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }
    double ur_tmp = ur;
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*(LI+LC+1)*(LJ+LC+1);
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur_tmp, prad+i*(LJ+LC+1)+j);
        }}
        const int ir = threadIdx.x;
        ur_tmp *= r128[ir];
    }
    __syncthreads();

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;

    __shared__ double angi[LI1*nfi*LIC1];
    __shared__ double angj[LJ1*nfj*LJC1];

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < 2*LC+1; m++){
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
            gctr[ij] += fac*s;
            //atomicAdd(gctr+ij, fac*s);
        }
        __syncthreads();
    }
}

template <int orderi, int orderj> __device__
void type2_cart_kernel(double *gctr,
                const int LI, const int LJ, const int LC,
                const int ish, const int jsh, const int ksh,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
    extern __shared__ double smem[];

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

    double* omegai = smem + (LI+LJ+1) * (LI+LC+1) * (LJ+LC+1);
    double* omegaj = omegai + (LI+LC+2)/2 * (LI+1)*(LI+2)*(LI+3)/6 * (2*LC+1);

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();

    double* rad_all = smem;
    set_shared_memory(rad_all, (LI+LJ+1)*(LI+LC+1)*(LJ+LC+1));

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    
    double radi[AO_LMAX+ECP_LMAX+orderi+1];
    type2_facs_rad<orderi>(radi, LI+LC, npi, dca, ci, ai);
    double radj[AO_LMAX+ECP_LMAX+orderj+1];
    type2_facs_rad<orderj>(radj, LJ+LC, npj, dcb, cj, aj);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }
    double ur_tmp = ur;
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*(LI+LC+1)*(LJ+LC+1);
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur_tmp, prad+i*(LJ+LC+1)+j);
        }}
        const int ir = threadIdx.x;
        ur_tmp *= r128[ir];
    }
    __syncthreads();

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    double* angi = omegaj + (LJ+LC+2)/2 * (LJ+1)*(LJ+2)*(LJ+3)/6 * (2*LC+1);
    double* angj = angi + (LI+1)*nfi*(LI+LC+1);

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }
    __syncthreads();

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

                double reg_angi[AO_LMAX+ECP_LMAX+orderi+1];
                double reg_angj[AO_LMAX+ECP_LMAX+orderj+1];
                for (int p = 0; p < LIC1; p++){reg_angi[p] = pangi[p];}
                for (int q = 0; q < LJC1; q++){reg_angj[q] = pangj[q];}
                for (int p = 0; p < LIC1; p++){
                for (int q = 0; q < LJC1; q++){
                    s += prad[p*LJC1+q] * reg_angi[p] * reg_angj[q];
                }}
            }}
            gctr[ij] += fac*s;
            //atomicAdd(gctr+ij, fac*s);
        }
        __syncthreads();
    }
}

// Unrolling case
template <int LI, int LJ, int LC> __global__
void type2_cart_ip1(double *gctr,
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

    constexpr int nfi1 = (LI+2) * (LI+3)/2;
    __shared__ double buf[nfi1*nfj];
    type2_cart_unrolled_kernel<1,0,LI+1,LJ,LC>(
        buf, ish, jsh, ksh, 
        ecpbas, ecploc, 
        atm, bas, env);
    _li_down(gctr_smem, buf, LI, LJ);
    __syncthreads();
    if constexpr (LI > 0){
        type2_cart_unrolled_kernel<0,0,LI-1,LJ,LC>(
            buf, ish, jsh, ksh, 
            ecpbas, ecploc, 
            atm, bas, env);
        _li_up(gctr_smem, buf, LI, LJ);
        __syncthreads();
    }

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gctr +   nao*nao;
        double *gz = gctr + 2*nao*nao;
        atomicAdd(gx + i*nao + j, gctr_smem[ij]);
        atomicAdd(gy + i*nao + j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz + i*nao + j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}


__global__
void type2_cart_ip1_general(double *gctr,
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
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += 3*ecp_id*nao*nao + ioff*nao + joff;

    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];
    for (int ij = threadIdx.x; ij < NF_MAX*NF_MAX*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int NFI_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int NFJ_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf[NFI_MAX*NFJ_MAX];
    type2_cart_kernel<1,0>(
        buf, LI+1, LJ, LC, 
        ish, jsh, ksh, 
        ecpbas, ecploc, 
        atm, bas, env);
    _li_down(gctr_smem, buf, LI, LJ);
    __syncthreads();
    if (LI > 0){
        type2_cart_kernel<0,0>(
            buf, LI-1, LJ, LC, 
            ish, jsh, ksh, 
            ecpbas, ecploc, 
            atm, bas, env);
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
        atomicAdd(gx + i*nao + j, gctr_smem[ij]);
        atomicAdd(gy + i*nao + j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz + i*nao + j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}
