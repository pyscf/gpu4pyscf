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

#ifdef USE_SYCL
#include "gint/sycl_device.hpp"
#endif // USE_SYCL

template <int orderi, int orderj> __device__
void type2_cart_kernel(double* smem, double *gctr,
                const int LI, const int LJ, const int LC,
                const int ish, const int jsh, const int ksh,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env)
{
#ifdef USE_SYCL
    auto item = sycl::ext::oneapi::experimental::this_nd_item<1>();
    const int threadIdx_x = item.get_local_id(0);
    const int blockDim_x = item.get_local_range(0);
#else // USE_SYCL
    const int threadIdx_x = threadIdx.x;
    const int blockDim_x = blockDim.x;
#endif // USE_SYCL

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

    double radi[AO_LMAX+ECP_LMAX+orderi+1];
    double radj[AO_LMAX+ECP_LMAX+orderj+1];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    type2_facs_rad<orderi>(radi, LI+LC, npi, dca, ci, ai);
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
        const int ir = threadIdx_x;
        ur_tmp *= r128[ir];
    }
    __syncthreads();

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    double* angi = omegaj + (LJ+LC+2)/2 * (LJ+1)*(LJ+2)*(LJ+3)/6 * (2*LC+1);
    double* angj = angi + (LI+1)*nfi*(LI+LC+1);

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx_x; ij < nfi*nfj; ij+=blockDim_x){
        gctr[ij] = 0.0;
    }
    __syncthreads();

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < 2*LC+1; m++){
        type2_ang(angi, LI, LC, rca, omegai+m);
        type2_ang(angj, LJ, LC, rcb, omegaj+m);
        __syncthreads();
        for (int ij = threadIdx_x; ij < nfi*nfj; ij+=blockDim_x){
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

__global__
void type2_cart_ip1(double *gctr,
                const int LI, const int LJ, const int LC,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env
#ifdef USE_SYCL
                , sycl::nd_item<1> &item, double* smem
#endif                    
                    )
{
    const int orderi = 1;
    const int orderj = 0;  
    constexpr int NFI_MAX = (AO_LMAX+orderi+1)*(AO_LMAX+orderi+2)/2;
    constexpr int NFJ_MAX = (AO_LMAX+orderj+1)*(AO_LMAX+orderj+2)/2;
  
#ifdef USE_SYCL
    const int task_id = item.get_group(0);
    const int threadIdx_x = item.get_local_id(0);
    const int blockDim_x = item.get_local_range(0);
    double (&gctr_smem)[NF_MAX*NF_MAX*3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[NF_MAX*NF_MAX*3]>(item.get_group());
    double (&buf)[NFI_MAX*NFJ_MAX] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[NFI_MAX*NFJ_MAX]>(item.get_group());    
#else // USE_SYCL
    const int task_id = blockIdx.x;
    const int threadIdx_x = threadIdx.x;
    const int blockDim_x = blockDim.x;
    extern __shared__ double smem[];
    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];
    __shared__ double buf[NFI_MAX*NFJ_MAX];    
#endif // USE_SYCL  
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

    for (int ij = threadIdx_x; ij < NF_MAX*NF_MAX*3; ij+=blockDim_x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    type2_cart_kernel<1,0>(smem, buf, LI+1, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    _li_down(gctr_smem, buf, LI, LJ);
    if (LI > 0){
        type2_cart_kernel<0,0>(smem, buf, LI-1, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        _li_up(gctr_smem, buf, LI, LJ);
    }

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    for (int ij = threadIdx_x; ij < nfi*nfj; ij+=blockDim_x){
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
void type2_cart_ipipv(double *gctr,
                const int LI, const int LJ, const int LC,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const int *atm, const int *bas, const double *env
#ifdef USE_SYCL
                , sycl::nd_item<1> &item, double* smem
#endif                      
                      )
{
    constexpr int nfi2_max = (AO_LMAX+3)*(AO_LMAX+4)/2;
    constexpr int nfj_max = (AO_LMAX+1)*(AO_LMAX+2)/2;  

#ifdef USE_SYCL
    const int task_id = item.get_group(0);
    using tile_t = double[nfi2_max*nfj_max];
    tile_t& buf1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(item.get_group());
#else
    const int task_id = blockIdx.x;
    __shared__ double buf1[nfi2_max*nfj_max];
    extern __shared__ double smem[];
#endif
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    type2_cart_kernel<2,0>(smem, buf1, LI+2, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();

    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    double *buf = smem;
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_down(buf, buf1, LI+1, LJ);
    _li_down_and_write(gctr, buf, LI, LJ, nao);

    type2_cart_kernel<1,0>(smem, buf1, LI, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_up(buf, buf1, LI+1, LJ);
    _li_down_and_write(gctr, buf, LI, LJ, nao);

    if (LI > 0){
        set_shared_memory(buf, 3*nfi1_max*nfj_max);
        _li_down(buf, buf1, LI-1, LJ);
        _li_up_and_write(gctr, buf, LI, LJ, nao);
        if (LI > 1){
            type2_cart_kernel<0,0>(smem, buf1, LI-2, LJ, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
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
                const int *atm, const int *bas, const double *env
#ifdef USE_SYCL
                , sycl::nd_item<1> &item, double* smem
#endif
                      )
{
    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;  
#ifdef USE_SYCL
    const int task_id = item.get_group(0);
    using tile_t = double[nfi1_max*nfj1_max];
    tile_t& buf1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(item.get_group());
#else
    const int task_id = blockIdx.x;        
    __shared__ double buf1[nfi1_max*nfj1_max];    
    extern __shared__ double smem[];
#endif
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    type2_cart_kernel<1,1>(smem, buf1, LI+1, LJ+1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();

    constexpr int nfi_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    double *buf = smem;
    set_shared_memory(buf, 3*nfi_max*nfj1_max);
    _li_down(buf, buf1, LI, LJ+1);
    _lj_down_and_write(gctr, buf, LI, LJ, nao);

    if (LI > 0){
        type2_cart_kernel<0,1>(smem, buf1, LI-1, LJ+1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_up(buf, buf1, LI, LJ+1);
        _lj_down_and_write(gctr, buf, LI, LJ, nao);
    }

    if (LJ > 0){
        type2_cart_kernel<1,0>(smem, buf1, LI+1, LJ-1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_down(buf, buf1, LI, LJ-1);
         _lj_up_and_write(gctr, buf, LI, LJ, nao);
        if (LI > 0){
            type2_cart_kernel<0,0>(smem, buf1, LI-1, LJ-1, LC, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
            __syncthreads();
            set_shared_memory(buf, 3*nfi_max*nfj1_max);
            _li_up(buf, buf1, LI, LJ-1);
            _lj_up_and_write(gctr, buf, LI, LJ, nao);
        }
    }
    return;
}
