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
void type2_cart_kernel(double *gctr, 
                const int LI, const int LJ, const int LC,
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
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    const int LI1 = LI+1;
    const int LJ1 = LJ+1;
    const int LIC1 = LI+LC+1;
    const int LJC1 = LJ+LC+1;
    const int LCC1 = (2*LC+1);

    double* rad_all = smem;
    for (int i = threadIdx.x; i < (LI+LJ+1)*LIC1*LJC1; i+=blockDim.x){
        rad_all[i] = 0.0;
    }
    __syncthreads();
    
    double radi[AO_LMAX+ECP_LMAX+orderi+1];
    double radj[AO_LMAX+ECP_LMAX+orderj+1];
    type2_facs_rad<orderi>(radi, LI+LC, npi, dca, ci, ai);
    type2_facs_rad<orderj>(radj, LJ+LC, npj, dcb, cj, aj);

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
        const int ir = threadIdx.x;
        ur_tmp *= r128[ir];
    }
    __syncthreads();

    const int BLKI = (LIC1+1)/2 * LCC1;
    const int BLKJ = (LJC1+1)/2 * LCC1;

    double* omegai = smem + (LI+LJ+1) * LIC1 * LJC1;
    double* omegaj = omegai + LI1*(LI1+1)*(LI1+2)/6 * BLKI;

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    double* angi = omegaj + LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ;
    double* angj = angi + LI1*nfi*LIC1;

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }
    __syncthreads();

    constexpr int NFI_MAX = (AO_LMAX+orderi+1)*(AO_LMAX+orderi+2)/2;
    constexpr int NFJ_MAX = (AO_LMAX+orderj+1)*(AO_LMAX+orderj+2)/2;
    double fi[3*NFI_MAX];
    double fj[3*NFJ_MAX];
    cache_fac(fi, LI, rca);
    cache_fac(fj, LJ, rcb);
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
                
                double reg_angi[AO_LMAX+ECP_LMAX+orderi+1];
                double reg_angj[AO_LMAX+ECP_LMAX+orderj+1];
                for (int p = 0; p < LIC1; p++){reg_angi[p] = pangi[p];}
                for (int q = 0; q < LJC1; q++){reg_angj[q] = pangj[q];}
                for (int p = 0; p < LIC1; p++){
                for (int q = 0; q < LJC1; q++){
                    s += prad[p*LJC1+q] * reg_angi[p] * reg_angj[q];
                }}
            }}
            //gctr[ij] += fac*s;
            atomicAdd(gctr+ij, fac*s);
        }
        __syncthreads();
    }
}

__global__
void type2_cart_ip1(double *gctr, 
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
    
    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];
    for (int ij = threadIdx.x; ij < NF_MAX*NF_MAX*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();
    
    const int orderi = 1;
    const int orderj = 0;
    constexpr int NFI_MAX = (AO_LMAX+orderi+1)*(AO_LMAX+orderi+2)/2;
    constexpr int NFJ_MAX = (AO_LMAX+orderj+1)*(AO_LMAX+orderj+2)/2;
    __shared__ double buf[NFI_MAX*NFJ_MAX];
    type2_cart_kernel<1,0>(buf, LI+1, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    _li_down(gctr_smem, buf, LI, LJ);
    if (LI > 0){
        type2_cart_kernel<0,0>(buf, LI-1, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        _li_up(gctr_smem, buf, LI, LJ);
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


__global__
void type2_cart_ipipv(double *gctr, 
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
    gctr += ioff*nao + joff;
    
    constexpr int nfi2_max = (AO_LMAX+3)*(AO_LMAX+4)/2;
    constexpr int nfj_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf1[nfi2_max*nfj_max];
    type2_cart_kernel<2,0>(buf1, LI+2, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    
    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    extern __shared__ double smem[];
    double *buf = smem;
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_down(buf, buf1, LI+1, LJ);
    _li_down_and_write(gctr, buf, LI, LJ, nao);

    type2_cart_kernel<1,0>(buf1, LI, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_up(buf, buf1, LI+1, LJ);
    _li_down_and_write(gctr, buf, LI, LJ, nao);

    if (LI > 0){
        set_shared_memory(buf, 3*nfi1_max*nfj_max);
        _li_down(buf, buf1, LI-1, LJ);
        _li_up_and_write(gctr, buf, LI, LJ, nao);
        if (LI > 1){
            type2_cart_kernel<0,0>(buf1, LI-2, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
            __syncthreads();
            set_shared_memory(buf, 3*nfi1_max*nfj_max);
            _li_up(buf, buf1, LI-1, LJ);
            _li_up_and_write(gctr, buf, LI, LJ, nao);
        }
    }
    return;
}

__global__
void type2_cart_ipvip(double *gctr, 
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
    gctr += ioff*nao + joff;
    
    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    __shared__ double buf1[nfi1_max*nfj1_max];
    type2_cart_kernel<1,1>(buf1, LI+1, LJ+1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();

    constexpr int nfi_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    extern __shared__ double smem[];
    double *buf = smem;
    set_shared_memory(buf, 3*nfi_max*nfj1_max);
    _li_down(buf, buf1, LI, LJ+1);
    _lj_down_and_write(gctr, buf, LI, LJ, nao);

    if (LI > 0){
        type2_cart_kernel<0,1>(buf1, LI-1, LJ+1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_up(buf, buf1, LI, LJ+1);
        _lj_down_and_write(gctr, buf, LI, LJ, nao);
    }
    
    if (LJ > 0){
        type2_cart_kernel<1,0>(buf1, LI+1, LJ-1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_down(buf, buf1, LI, LJ-1);
         _lj_up_and_write(gctr, buf, LI, LJ, nao);
        if (LI > 0){
            type2_cart_kernel<0,0>(buf1, LI-1, LJ-1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
            __syncthreads();
            set_shared_memory(buf, 3*nfi_max*nfj1_max);
            _li_up(buf, buf1, LI, LJ-1);
            _lj_up_and_write(gctr, buf, LI, LJ, nao);
        }
    }
    return;
}
