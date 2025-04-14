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
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    constexpr int nfi2_max = (AO_LMAX+3)*(AO_LMAX+4)/2;
    constexpr int nfj_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf1[nfi2_max*nfj_max];
    type2_cart_kernel<2,0>(
        buf1, LI+2, LJ, LC,
        ish, jsh, ksh,
        ecpbas, ecploc,
        atm, bas, env);

    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    extern __shared__ double smem[];
    double *buf = smem;
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_down(buf, buf1, LI+1, LJ);
    __syncthreads();
    _li_down_and_write(gctr, buf, LI, LJ, nao);
    __syncthreads();

    type2_cart_kernel<1,0>(
        buf1, LI, LJ, LC,
        ish, jsh, ksh,
        ecpbas, ecploc,
        atm, bas, env);
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_up(buf, buf1, LI+1, LJ);
    __syncthreads();
    _li_down_and_write(gctr, buf, LI, LJ, nao);
    __syncthreads();

    if (LI > 0){
        set_shared_memory(buf, 3*nfi1_max*nfj_max);
        _li_down(buf, buf1, LI-1, LJ);
        __syncthreads();
        _li_up_and_write(gctr, buf, LI, LJ, nao);
        __syncthreads();
        if (LI > 1){
            type2_cart_kernel<0,0>(
                buf1, LI-2, LJ, LC,
                ish, jsh, ksh,
                ecpbas, ecploc,
                atm, bas, env);
            set_shared_memory(buf, 3*nfi1_max*nfj_max);
            _li_up(buf, buf1, LI-1, LJ);
            __syncthreads();
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
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    __shared__ double buf1[nfi1_max*nfj1_max];
    type2_cart_kernel<1,1>(
        buf1, LI+1, LJ+1, LC,
        ish, jsh, ksh,
        ecpbas, ecploc,
        atm, bas, env);

    constexpr int nfi_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    extern __shared__ double smem[];
    double *buf = smem;
    set_shared_memory(buf, 3*nfi_max*nfj1_max);
    _li_down(buf, buf1, LI, LJ+1);
    __syncthreads();
    _lj_down_and_write(gctr, buf, LI, LJ, nao);
    __syncthreads();
    if (LI > 0){
        type2_cart_kernel<0,1>(
            buf1, LI-1, LJ+1, LC,
            ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env);
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_up(buf, buf1, LI, LJ+1);
        __syncthreads();
        _lj_down_and_write(gctr, buf, LI, LJ, nao);
        __syncthreads();
    }

    if (LJ > 0){
        type2_cart_kernel<1,0>(
            buf1, LI+1, LJ-1, LC,
            ish, jsh, ksh,
            ecpbas, ecploc,
            atm, bas, env);
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_down(buf, buf1, LI, LJ-1);
        __syncthreads();
        _lj_up_and_write(gctr, buf, LI, LJ, nao);
        __syncthreads();
        if (LI > 0){
            type2_cart_kernel<0,0>(
                buf1, LI-1, LJ-1, LC,
                ish, jsh, ksh,
                ecpbas, ecploc,
                atm, bas, env);
            set_shared_memory(buf, 3*nfi_max*nfj1_max);
            _li_up(buf, buf1, LI, LJ-1);
            __syncthreads();
            _lj_up_and_write(gctr, buf, LI, LJ, nao);
            __syncthreads();
        }
    }
    return;
}
