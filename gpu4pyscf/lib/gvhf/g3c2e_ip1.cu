/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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


template <int LI, int LJ, int LK> __device__
static void GINTkernel_int3c2e_ip1_getjk_direct(GINTEnvVars envs, JKMatrix jk, double* j3, double* k3, 
        double* g, const double ai2,
        const int ish, const int jsh, const int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;

    constexpr int LI_CEIL = LI + 1;
    constexpr int NROOTS = (LI_CEIL+LJ+LK)/2 + 1;
    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2;

    constexpr int di = NROOTS;
    constexpr int dj = di * (LI_CEIL + 1);
    constexpr int dk = dj * (LJ + 1);
    constexpr int g_size = dk * (LK + 1);

    const int nao = jk.nao;
    
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;

    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    if (rhoj == NULL){
        for (int kp = 0; kp < nfk; ++kp) {
        for (int jp = 0; jp < nfj; ++jp) {
        for (int ip = 0; ip < nfi; ++ip) { 
            const int loc_k = c_l_locs[LK] + kp;
            const int loc_j = c_l_locs[LJ] + jp;
            const int loc_i = c_l_locs[LI] + ip;

            const int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
            const int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
            const int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;
            
            const int i_idx = idx[loc_i];
            const int i_idy = idy[loc_i];
            const int i_idz = idz[loc_i];

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < NROOTS; ++ir){
                const double gx = g[ix+ir];
                const double gy = g[iy+ir];
                const double gz = g[iz+ir];

                double fx = ai2*g[ix+ir+di];
                double fy = ai2*g[iy+ir+di];
                double fz = ai2*g[iz+ir+di];

                fx += i_idx>0 ? i_idx*g[ix+ir-di] : 0.0;
                fy += i_idy>0 ? i_idy*g[iy+ir-di] : 0.0;
                fz += i_idz>0 ? i_idz*g[iz+ir-di] : 0.0;

                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }

            const int off_rhok = (i0+ip) + nao*(j0+jp) + (k0+kp)*nao*nao;
            const double rhok_tmp = rhok[off_rhok];
            k3[0] += rhok_tmp * sx;
            k3[1] += rhok_tmp * sy;
            k3[2] += rhok_tmp * sz;
        }}}
        return;
    }

    if (rhok == NULL){
        for (int ip = 0; ip < nfi; ++ip) {
        for (int jp = 0; jp < nfj; ++jp) {
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            for (int kp = 0; kp < nfk; ++kp) {
                const int loc_j = c_l_locs[LJ] + jp;
                const int loc_i = c_l_locs[LI] + ip;
                const int loc_k = c_l_locs[LK] + kp;
                
                const int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
                const int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                const int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                const int i_idx = idx[loc_i];
                const int i_idy = idy[loc_i];
                const int i_idz = idz[loc_i];

                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
#pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    const double gx = g[ix+ir];
                    const double gy = g[iy+ir];
                    const double gz = g[iz+ir];
                    
                    double fx = ai2*g[ix+ir+di];
                    double fy = ai2*g[iy+ir+di];
                    double fz = ai2*g[iz+ir+di];

                    fx += i_idx>0 ? i_idx*g[ix+ir-di] : 0.0;
                    fy += i_idy>0 ? i_idy*g[iy+ir-di] : 0.0;
                    fz += i_idz>0 ? i_idz*g[iz+ir-di] : 0.0;

                    sx += fx * gy * gz;
                    sy += gx * fy * gz;
                    sz += gx * gy * fz;
                }

                const double rhoj_k = rhoj[kp + k0];
                jx += rhoj_k * sx;
                jy += rhoj_k * sy;
                jz += rhoj_k * sz;
            }
            const int off_dm = (ip + i0) + nao*(j0 + jp);
            const double dm_ij = dm[off_dm];
            j3[0] += jx * dm_ij;
            j3[1] += jy * dm_ij;
            j3[2] += jz * dm_ij;
        }}
        return;
    }

    for (int ip = 0; ip < nfi; ++ip) {
    for (int jp = 0; jp < nfj; ++jp) {
        double jx = 0.0;
        double jy = 0.0;
        double jz = 0.0;
        for (int kp = 0; kp < nfk; ++kp) {                
            const int loc_k = c_l_locs[LK] + kp;
            const int loc_j = c_l_locs[LJ] + jp;
            const int loc_i = c_l_locs[LI] + ip;
            
            const int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
            const int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
            const int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;
            
            const int i_idx = idx[loc_i];
            const int i_idy = idy[loc_i];
            const int i_idz = idz[loc_i];

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < NROOTS; ++ir){
                const double gx = g[ix+ir];
                const double gy = g[iy+ir];
                const double gz = g[iz+ir];

                double fx = ai2*g[ix+ir+di];
                double fy = ai2*g[iy+ir+di];
                double fz = ai2*g[iz+ir+di];

                fx += i_idx>0 ? i_idx*g[ix+ir-di] : 0.0;
                fy += i_idy>0 ? i_idy*g[iy+ir-di] : 0.0;
                fz += i_idz>0 ? i_idz*g[iz+ir-di] : 0.0;

                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }

            const int off_rhok = (i0 + ip) + nao*(j0 + jp) + (k0 + kp)*nao*nao;
            const double rhok_tmp = rhok[off_rhok];
            k3[0] += rhok_tmp * sx;
            k3[1] += rhok_tmp * sy;
            k3[2] += rhok_tmp * sz;

            const double rhoj_k = rhoj[k0+kp];
            jx += rhoj_k * sx;
            jy += rhoj_k * sy;
            jz += rhoj_k * sz;
        }
        const int off_dm = (i0 + ip) + nao*(j0 + jp);
        const double dm_ij = dm[off_dm];
        j3[0] += jx * dm_ij;
        j3[1] += jy * dm_ij;
        j3[2] += jz * dm_ij;
    }}
}

__device__
static void write_int3c2e_ip1_jk(JKMatrix jk, double* j3, double* k3, int ish){
    int *bas_atm = c_bpcache.bas_atm;
    const int atm_id = bas_atm[ish];
    double *vj = jk.vj;
    double *vk = jk.vk;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];
    
    if (vj != NULL){
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = j3[j]; __syncthreads();
            if(THREADSX >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSX >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSX >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSX >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (ty == 0) atomicAdd(vj + 3*atm_id+j, sdata[tx][0]);
        }
    }

    if (vk != NULL){
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = k3[j]; __syncthreads();
            if(THREADSX >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSX >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSX >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSX >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (ty == 0) atomicAdd(vk + 3*atm_id+j, sdata[tx][0]);
        }
    }
}

// Unrolled version
template <int LI, int LJ, int LK> __global__
void GINTint3c2e_ip1_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int bas_kl = offsets.bas_kl + task_kl;
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];
    
    double* __restrict__ exp = c_bpcache.a1;
    constexpr int LI_CEIL = LI + 1;
    constexpr int NROOTS = (LI_CEIL+LJ+LK)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI_CEIL+1)*(LJ+1)*(LK+1);

    double g[GSIZE];

    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    double j3[3] = {0.0};
    double k3[3] = {0.0};

    if (active) {
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_int3c2e<LI_CEIL, LJ, LK>(envs, g, as_ish, as_jsh, ksh, ij, kl);
            const double ai2 = -2.0*exp[ij];
            GINTkernel_int3c2e_ip1_getjk_direct<LI, LJ, LK>(envs, jk, j3, k3, g, ai2, ish, jsh, ksh);
        }}
    }
    write_int3c2e_ip1_jk(jk, j3, k3, ish);
}


__device__
static void GINTkernel_int3c2e_ip1_getjk_direct(GINTEnvVars envs, JKMatrix jk, 
        double* j3, double* k3, double* g, double ai2,
        const int ish, const int jsh, const int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    const int i1 = ao_loc[ish+1] - jk.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    const int j1 = ao_loc[jsh+1] - jk.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    const int k1 = ao_loc[ksh+1] - jk.ao_offsets_k;
    const int di = envs.stride_i;
    const int dj = envs.stride_j;
    const int dk = envs.stride_k;
    const int g_size = envs.g_size;
    const int nao = jk.nao;

    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ dm = jk.dm;

    const int i_l = envs.i_l;
    const int j_l = envs.j_l;
    const int k_l = envs.k_l;
    const int nrys_roots = envs.nrys_roots;

    int *idx = c_idx;
    int *idy = c_idx + TOT_NF;
    int *idz = c_idx + TOT_NF * 2;

    if (rhoj == NULL){
        for (int tx = threadIdx.x; tx < (k1-k0)*(j1-j0)*(i1-i0); tx += blockDim.x) {
            const int kp = tx / ((j1-j0)*(i1-i0));
            const int jp = (tx / (i1-i0)) % (j1-j0);
            const int ip = tx % (i1-i0);
            
            const int loc_k = c_l_locs[k_l] + kp;
            const int loc_j = c_l_locs[j_l] + jp;
            const int loc_i = c_l_locs[i_l] + ip;

            const int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
            const int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
            const int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

            const int i_idx = idx[loc_i];
            const int i_idy = idy[loc_i];
            const int i_idz = idz[loc_i];

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < nrys_roots; ++ir){
                const double gx = g[ix+ir];
                const double gy = g[iy+ir];
                const double gz = g[iz+ir];

                double fx = ai2*g[ix+ir+di];
                double fy = ai2*g[iy+ir+di];
                double fz = ai2*g[iz+ir+di];

                fx += i_idx>0 ? i_idx*g[ix+ir-di] : 0.0;
                fy += i_idy>0 ? i_idy*g[iy+ir-di] : 0.0;
                fz += i_idz>0 ? i_idz*g[iz+ir-di] : 0.0;
                
                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }

            int off_rhok = (ip+i0) + nao*(jp+j0) + (kp+k0)*nao*nao;
            double rhok_tmp = rhok[off_rhok];
            k3[0] += rhok_tmp * sx;
            k3[1] += rhok_tmp * sy;
            k3[2] += rhok_tmp * sz;
        }
        return;
    }

    if (rhok == NULL){
        for (int tx = threadIdx.x; tx < (k1-k0)*(j1-j0)*(i1-i0); tx += blockDim.x) {
            const int kp = tx / ((j1-j0)*(i1-i0));
            const int jp = (tx / (i1-i0)) % (j1-j0);
            const int ip = tx % (i1-i0);
            
            const int loc_j = c_l_locs[j_l] + jp;
            const int loc_i = c_l_locs[i_l] + ip;
            const int loc_k = c_l_locs[k_l] + kp;
            
            const int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
            const int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
            const int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

            const int i_idx = idx[loc_i];
            const int i_idy = idy[loc_i];
            const int i_idz = idz[loc_i];

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < nrys_roots; ++ir){
                const double gx = g[ix+ir];
                const double gy = g[iy+ir];
                const double gz = g[iz+ir];
                
                double fx = ai2*g[ix+ir+di];
                double fy = ai2*g[iy+ir+di];
                double fz = ai2*g[iz+ir+di];

                fx += i_idx>0 ? i_idx*g[ix+ir-di] : 0.0;
                fy += i_idy>0 ? i_idy*g[iy+ir-di] : 0.0;
                fz += i_idz>0 ? i_idz*g[iz+ir-di] : 0.0;
                
                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }

            const int off_dm = (ip+i0) + nao*(jp+j0);
            const double rhoj_dm = dm[off_dm] * rhoj[kp+k0];
            j3[0] += rhoj_dm * sx;
            j3[1] += rhoj_dm * sy;
            j3[2] += rhoj_dm * sz;
        }
        return;
    }

    for (int tx = threadIdx.x; tx < (k1-k0)*(j1-j0)*(i1-i0); tx += blockDim.x) {
        const int kp = tx / ((j1-j0)*(i1-i0));
        const int jp = (tx / (i1-i0)) % (j1-j0);
        const int ip = tx % (i1-i0);

        const int loc_k = c_l_locs[k_l] + kp;
        const int loc_j = c_l_locs[j_l] + jp;
        const int loc_i = c_l_locs[i_l] + ip;

        const int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
        const int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
        const int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

        const int i_idx = idx[loc_i];
        const int i_idy = idy[loc_i];
        const int i_idz = idz[loc_i];

        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
#pragma unroll
        for (int ir = 0; ir < nrys_roots; ++ir){
            const double gx = g[ix+ir];
            const double gy = g[iy+ir];
            const double gz = g[iz+ir];

            double fx = ai2*g[ix+ir+di];
            double fy = ai2*g[iy+ir+di];
            double fz = ai2*g[iz+ir+di];
            
            fx += i_idx>0 ? i_idx*g[ix+ir-di] : 0.0;
            fy += i_idy>0 ? i_idy*g[iy+ir-di] : 0.0;
            fz += i_idz>0 ? i_idz*g[iz+ir-di] : 0.0;

            sx += fx * gy * gz;
            sy += gx * fy * gz;
            sz += gx * gy * fz;
        }

        const int off_rhok = (ip+i0) + nao*(jp+j0) + (kp+k0)*nao*nao;
        const double rhok_tmp = rhok[off_rhok];
        k3[0] += rhok_tmp * sx;
        k3[1] += rhok_tmp * sy;
        k3[2] += rhok_tmp * sz;

        const int off_dm = (ip+i0) + nao*(jp+j0);
        const double rhoj_dm = dm[off_dm] * rhoj[kp+k0];
        j3[0] += rhoj_dm * sx;
        j3[1] += rhoj_dm * sy;
        j3[2] += rhoj_dm * sz;
    }
}

__global__
void GINTint3c2e_ip1_jk_general_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    const int task_ij = blockIdx.x;// * blockDim.x + threadIdx.x;
    const int task_kl = blockIdx.y;// * blockDim.y + threadIdx.y;
    const int bas_ij = offsets.bas_ij + task_ij;
    const int bas_kl = offsets.bas_kl + task_kl;
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];
    
    extern __shared__ double g[];

    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    double j3[3] = {0.0};
    double k3[3] = {0.0};

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_int3c2e_shared(envs, g, as_ish, as_jsh, ksh, ij, kl);
        double ai2 = -2.0*c_bpcache.a1[ij];
        GINTkernel_int3c2e_ip1_getjk_direct(envs, jk, j3, k3, g, ai2, ish, jsh, ksh);
    }}
    
    constexpr int nthreads = THREADSX * THREADSY;
    int *bas_atm = c_bpcache.bas_atm;
    int atm_id = bas_atm[ish];
    if (jk.vj != NULL){
        block_reduce<nthreads>(jk.vj+3*atm_id,   j3[0]);
        block_reduce<nthreads>(jk.vj+3*atm_id+1, j3[1]);
        block_reduce<nthreads>(jk.vj+3*atm_id+2, j3[2]);
    }
    if (jk.vk != NULL){
        block_reduce<nthreads>(jk.vk+3*atm_id,   k3[0]);
        block_reduce<nthreads>(jk.vk+3*atm_id+1, k3[1]);
        block_reduce<nthreads>(jk.vk+3*atm_id+2, k3[2]);
    }
}

__global__
static void GINTint3c2e_ip1_jk_kernel000(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    bool active = true;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        active = false;
        task_ij = 0;
        task_kl = 0;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int bas_kl = offsets.bas_kl + task_kl;
    const double norm = envs.fac;
    const double omega = envs.omega;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];
    const int ksh = bas_pair2bra[bas_kl];
    const int nprim_ij = envs.nprim_ij;
    const int nprim_kl = envs.nprim_kl;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    double* __restrict__ a12 = c_bpcache.a12;
    double* __restrict__ e12 = c_bpcache.e12;
    double* __restrict__ x12 = c_bpcache.x12;
    double* __restrict__ y12 = c_bpcache.y12;
    double* __restrict__ z12 = c_bpcache.z12;
    double* __restrict__ a1 = c_bpcache.a1;
    int ij, kl;
    int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
    const int nbas = c_bpcache.nbas;
    double* __restrict__ bas_x = c_bpcache.bas_coords;
    double* __restrict__ bas_y = bas_x + nbas;
    double* __restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    const double xi = bas_x[ish];
    const double yi = bas_y[ish];
    const double zi = bas_z[ish];
    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
        const double ai2 = -2.0*a1[ij];
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double xij = x12[ij];
        const double yij = y12[ij];
        const double zij = z12[ij];
        const double akl = a12[kl];
        const double ekl = e12[kl];
        const double xkl = x12[kl];
        const double ykl = y12[kl];
        const double zkl = z12[kl];
        const double xijxkl = xij - xkl;
        const double yijykl = yij - ykl;
        const double zijzkl = zij - zkl;
        const double aijkl = aij + akl;
        const double a1 = aij * akl;
        double a0 = a1 / aijkl;
        const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
        a0 *= theta;
        const double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
        const double fac = eij * ekl * sqrt(a0 / (a1 * a1 * a1));
        double root0, weight0;
        if (x < 3.e-7) {
            root0 = 0.5;
            weight0 = 1.;
        } else {
            const double tt = sqrt(x);
            const double fmt0 = SQRTPIE4 / tt * erf(tt);
            weight0 = fmt0;
            const double e = exp(-x);
            const double b = .5 / x;
            const double fmt1 = b * (fmt0 - e);
            root0 = fmt1 / (fmt0 - fmt1);
        }
        root0 /= root0 + 1 - root0 * theta;
        const double u2 = a0 * root0;
        const double tmp2 = akl * u2 / (u2 * aijkl + a1);;
        const double c00x = xij - xi - tmp2 * xijxkl;
        const double c00y = yij - yi - tmp2 * yijykl;
        const double c00z = zij - zi - tmp2 * zijzkl;
        const double g_0 = 1;
        const double g_1 = c00x;
        const double g_2 = 1;
        const double g_3 = c00y;
        const double g_4 = norm * fac * weight0;
        const double g_5 = g_4 * c00z;

        const double f_1 = ai2 * g_1;
        const double f_3 = ai2 * g_3;
        const double f_5 = ai2 * g_5;

        gout0 += f_1 * g_2 * g_4;
        gout1 += g_0 * f_3 * g_4;
        gout2 += g_0 * g_2 * f_5;
    } }

    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - jk.ao_offsets_i;
    const int j0 = ao_loc[jsh] - jk.ao_offsets_j;
    const int k0 = ao_loc[ksh] - jk.ao_offsets_k;

    const int nao = jk.nao;
    double* __restrict__ dm = jk.dm;
    double* __restrict__ rhok = jk.rhok;
    double* __restrict__ rhoj = jk.rhoj;
    double* __restrict__ vj = jk.vj;
    double* __restrict__ vk = jk.vk;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];
    if (!active){
        gout0 = 0.0; gout1 = 0.0; gout2 = 0.0;
    }

    int *bas_atm = c_bpcache.bas_atm;
    int atm_id = bas_atm[ish];
    if (vj != NULL){
        int off_dm = i0 + nao*j0;
        double rhoj_tmp = dm[off_dm] * rhoj[k0];
        double vj_tmp[3];
        vj_tmp[0] = gout0*rhoj_tmp;
        vj_tmp[1] = gout1*rhoj_tmp;
        vj_tmp[2] = gout2*rhoj_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vj_tmp[j]; __syncthreads();
            if(THREADSX >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSX >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSX >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSX >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (ty == 0) atomicAdd(vj + 3*atm_id+j, sdata[tx][0]);
        }
    }
    if (vk != NULL){
        int off_rhok = i0 + nao*j0 + k0*nao*nao;
        double rhok_tmp = rhok[off_rhok];
        double vk_tmp[3];
        vk_tmp[0] = gout0 * rhok_tmp;
        vk_tmp[1] = gout1 * rhok_tmp;
        vk_tmp[2] = gout2 * rhok_tmp;
        for (int j = 0; j < 3; j++){
            sdata[tx][ty] = vk_tmp[j]; __syncthreads();
            if(THREADSY >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSY >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSY >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSY >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (ty == 0) atomicAdd(vk + 3*atm_id+j, sdata[tx][0]);
        }
    }
}

