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
void type1_cart_kernel(double* smem, double *gctr,
                const int LI, const int LJ,
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
    for (int i = threadIdx_x; i < LIJ3; i+=blockDim_x) {
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

    constexpr int NFI_MAX = (AO_LMAX+orderi+1)*(AO_LMAX+orderi+2)/2;
    constexpr int NFJ_MAX = (AO_LMAX+orderj+1)*(AO_LMAX+orderj+2)/2;
    double fi[3*NFI_MAX];
    double fj[3*NFJ_MAX];
    cache_fac(fi, LI, rca);
    cache_fac(fj, LJ, rcb);

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    for (int ij = threadIdx_x; ij < nfi*nfj; ij+=blockDim_x){
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
            for (int j1 = 0; j1 <= jx; j1++){
            for (int j2 = 0; j2 <= jy; j2++){
            for (int j3 = 0; j3 <= jz; j3++){
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
                const int *atm, const int *bas, const double *env
#ifdef USE_SYCL
                , sycl::nd_item<1> &item, double* smem
#endif
                    )
{
    constexpr int nfi_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj_max = (AO_LMAX+1)*(AO_LMAX+2)/2;

#ifdef USE_SYCL
    const int task_id = item.get_group(0);
    const int threadIdx_x = item.get_local_id(0);
    const int blockDim_x = item.get_local_range(0);
    double (&buf)[nfi_max*nfj_max] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[nfi_max*nfj_max]>(item.get_group());
    double (&gctr_smem)[NF_MAX*NF_MAX*3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[NF_MAX*NF_MAX*3]>(item.get_group());    
#else // USE_SYCL
    const int task_id = blockIdx.x;
    const int threadIdx_x = threadIdx.x;
    const int blockDim_x = blockDim.x;
    extern __shared__ double smem[];
    __shared__ double buf[nfi_max*nfj_max];
    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];    
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

    type1_cart_kernel<1,0>(smem, buf, LI+1, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    _li_down(gctr_smem, buf, LI, LJ);

    if (LI > 0){
        type1_cart_kernel<0,0>(smem, buf, LI-1, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
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
        atomicAdd(gx+i*nao+j, gctr_smem[ij]);
        atomicAdd(gy+i*nao+j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz+i*nao+j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}

__global__
void type1_cart_ipipv(double *gctr,
                const int LI, const int LJ,
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
    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;

#ifdef USE_SYCL
    const int task_id = item.get_group(0);
    const int threadIdx_x = item.get_local_id(0);
    const int blockDim_x = item.get_local_range(0);

    double (&buf1)[nfi2_max*nfj_max] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[nfi2_max*nfj_max]>(item.get_group());
    double (&buf)[3*nfi1_max*nfj_max] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3*nfi1_max*nfj_max]>(item.get_group());
#else // USE_SYCL
    const int task_id = blockIdx.x;
    const int threadIdx_x = threadIdx.x;
    const int blockDim_x = blockDim.x;

    extern __shared__ double smem[];
    __shared__ double buf1[nfi2_max*nfj_max];
    __shared__ double buf[3*nfi1_max*nfj_max];

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
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    type1_cart_kernel<2,0>(smem, buf1, LI+2, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();

    for (int i = threadIdx_x; i < 3*nfi1_max*nfj_max; i+=blockDim_x){
        buf[i] = 0.0;
    }
    __syncthreads();
    _li_down(buf, buf1, LI+1, LJ);

    type1_cart_kernel<1,0>(smem, buf1, LI, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();
    _li_up(buf, buf1, LI+1, LJ);
    _li_down_and_write(gctr, buf, LI, LJ, nao);

    if (LI > 0){
        for (int i = threadIdx_x; i < 3*nfi1_max*nfj_max; i+=blockDim_x){
            buf[i] = 0.0;
        }
        __syncthreads();
        _li_down(buf, buf1, LI-1, LJ);
        if (LI > 1){
            type1_cart_kernel<0,0>(smem, buf1, LI-2, LJ, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
            __syncthreads();
            _li_up(buf, buf1, LI-1, LJ);
        }
        _li_up_and_write(gctr, buf, LI, LJ, nao);
    }
    return;
}

__global__
void type1_cart_ipvip(double *gctr,
                const int LI, const int LJ,
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
    constexpr int nfi_max = (AO_LMAX+1)*(AO_LMAX+2)/2;

#ifdef USE_SYCL
    const int task_id = item.get_group(0);
    const int threadIdx_x = item.get_local_id(0);
    const int blockDim_x = item.get_local_range(0);

    double (&buf1)[nfi1_max*nfj1_max] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[nfi1_max*nfj1_max]>(item.get_group());
    double (&buf)[3*nfi_max*nfj1_max] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3*nfi_max*nfj1_max]>(item.get_group());
#else
    const int task_id = blockIdx.x;
    const int threadIdx_x = threadIdx.x;
    const int blockDim_x = blockDim.x;

    __shared__ double buf1[nfi1_max*nfj1_max];
    __shared__ double buf[3*nfi_max*nfj1_max];
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

    type1_cart_kernel<1,1>(smem, buf1, LI+1, LJ+1, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
    __syncthreads();

    for (int i = threadIdx_x; i < 3*nfi_max*nfj1_max; i+=blockDim_x){
        buf[i] = 0.0;
    }
    __syncthreads();
    _li_down(buf, buf1, LI, LJ+1);
    if (LI > 0){
        type1_cart_kernel<0,1>(smem, buf1, LI-1, LJ+1, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        _li_up(buf, buf1, LI, LJ+1);
    }
    _lj_down_and_write(gctr, buf, LI, LJ, nao);

    if (LJ > 0){
        for (int i = threadIdx_x; i < 3*nfi_max*nfj1_max; i+=blockDim_x){
            buf[i] = 0.0;
        }
        __syncthreads();
        type1_cart_kernel<1,0>(smem, buf1, LI+1, LJ-1, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
        __syncthreads();
        _li_down(buf, buf1, LI, LJ-1);
        if (LI > 0){
            type1_cart_kernel<0,0>(smem, buf1, LI-1, LJ-1, ish, jsh, ksh, ecpbas, ecploc, atm, bas, env);
            __syncthreads();
            _li_up(buf, buf1, LI, LJ-1);
        }
        _lj_up_and_write(gctr, buf, LI, LJ, nao);
    }
    return;
}
