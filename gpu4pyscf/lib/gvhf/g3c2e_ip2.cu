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
static void GINTkernel_int3c2e_ip2_getjk_direct(GINTEnvVars envs, JKMatrix jk, 
        double* j3, double* k3, double *g, double ak2,
        int ish, int jsh, int ksh)
{
    int *ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish  ] - jk.ao_offsets_i;
    const int j0 = ao_loc[jsh  ] - jk.ao_offsets_j;
    const int k0 = ao_loc[ksh  ] - jk.ao_offsets_k;
    
    constexpr int LK_CEIL = LK + 1;
    constexpr int NROOTS = (LI+LJ+LK_CEIL)/2 + 1;
    constexpr int nfi = (LI+1)*(LI+2)/2;
    constexpr int nfj = (LJ+1)*(LJ+2)/2;
    constexpr int nfk = (LK+1)*(LK+2)/2; 

    constexpr int di = NROOTS;
    constexpr int dj = di * (LI + 1);
    constexpr int dk = dj * (LJ + 1);
    constexpr int g_size = dk * (LK_CEIL + 1);

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

            const int k_idx = idx[loc_k];
            const int k_idy = idy[loc_k];
            const int k_idz = idz[loc_k];

            const int ix = dk * k_idx + dj * idx[loc_j] + di * idx[loc_i];
            const int iy = dk * k_idy + dj * idy[loc_j] + di * idy[loc_i] + g_size;
            const int iz = dk * k_idz + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < NROOTS; ++ir){
                const double gx = g[ix+ir];
                const double gy = g[iy+ir];
                const double gz = g[iz+ir];

                double fx = ak2*g[ix+ir+dk];
                double fy = ak2*g[iy+ir+dk];
                double fz = ak2*g[iz+ir+dk];

                fx += k_idx>0 ? k_idx*g[ix+ir-dk] : 0.0;
                fy += k_idy>0 ? k_idy*g[iy+ir-dk] : 0.0;
                fz += k_idz>0 ? k_idz*g[iz+ir-dk] : 0.0;

                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }

            const int off_rhok = (i0+ip) + nao*(j0+jp) + (k0+kp)*nao*nao;
            const double rhok_tmp = rhok[off_rhok];
            k3[0] += sx * rhok_tmp;
            k3[1] += sy * rhok_tmp;
            k3[2] += sz * rhok_tmp;
        }}}
        return;
    }

    if (rhok == NULL){
        for (int kp = 0; kp < nfk; ++kp) {
            double jx = 0.0;
            double jy = 0.0;
            double jz = 0.0;
            
            for (int jp = 0; jp < nfj; ++jp) {
            for (int ip = 0; ip < nfi; ++ip) {
                const int loc_k = c_l_locs[LK] + kp;
                const int loc_j = c_l_locs[LJ] + jp;
                const int loc_i = c_l_locs[LI] + ip;
                
                const int k_idx = idx[loc_k];
                const int k_idy = idy[loc_k];
                const int k_idz = idz[loc_k];

                const int ix = dk * k_idx + dj * idx[loc_j] + di * idx[loc_i];
                const int iy = dk * k_idy + dj * idy[loc_j] + di * idy[loc_i] + g_size;
                const int iz = dk * k_idz + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

                double sx = 0.0;
                double sy = 0.0;
                double sz = 0.0;
    #pragma unroll
                for (int ir = 0; ir < NROOTS; ++ir){
                    const double gx = g[ix+ir];
                    const double gy = g[iy+ir];
                    const double gz = g[iz+ir];
                    
                    double fx = ak2*g[ix+ir+dk];
                    double fy = ak2*g[iy+ir+dk];
                    double fz = ak2*g[iz+ir+dk];

                    fx += k_idx>0 ? k_idx*g[ix+ir-dk] : 0.0;
                    fy += k_idy>0 ? k_idy*g[iy+ir-dk] : 0.0;
                    fz += k_idz>0 ? k_idz*g[iz+ir-dk] : 0.0;

                    sx += fx * gy * gz;
                    sy += gx * fy * gz;
                    sz += gx * gy * fz;
                }
                const int off_dm = (ip+i0) + nao*(jp+j0);
                const double dm_ij = dm[off_dm];
                jx += dm_ij * sx;
                jy += dm_ij * sy;
                jz += dm_ij * sz;
            }}
            const double rhoj_k = rhoj[kp + k0];
            j3[0] += jx * rhoj_k;
            j3[1] += jy * rhoj_k;
            j3[2] += jz * rhoj_k;
        }
        return;
    }

    for (int kp = 0; kp < nfk; ++kp) {
        double jx = 0.0;
        double jy = 0.0;
        double jz = 0.0;
        
        for (int jp = 0; jp < nfj; ++jp) {
        for (int ip = 0; ip < nfi; ++ip) {
            const int loc_k = c_l_locs[LK] + kp;
            const int loc_j = c_l_locs[LJ] + jp;
            const int loc_i = c_l_locs[LI] + ip;
            
            const int k_idx = idx[loc_k];
            const int k_idy = idy[loc_k];
            const int k_idz = idz[loc_k];

            const int ix = dk * k_idx + dj * idx[loc_j] + di * idx[loc_i];
            const int iy = dk * k_idy + dj * idy[loc_j] + di * idy[loc_i] + g_size;
            const int iz = dk * k_idz + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < NROOTS; ++ir){
                const double gx = g[ix+ir];
                const double gy = g[iy+ir];
                const double gz = g[iz+ir];

                double fx = ak2*g[ix+ir+dk];
                double fy = ak2*g[iy+ir+dk];
                double fz = ak2*g[iz+ir+dk];

                fx += k_idx>0 ? k_idx*g[ix+ir-dk] : 0.0;
                fy += k_idy>0 ? k_idy*g[iy+ir-dk] : 0.0;
                fz += k_idz>0 ? k_idz*g[iz+ir-dk] : 0.0;

                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }

            const int off_rhok = (i0+ip) + nao*(j0+jp) + (k0+kp)*nao*nao;
            const double rhok_tmp = rhok[off_rhok];
            k3[0] += sx * rhok_tmp;
            k3[1] += sy * rhok_tmp;
            k3[2] += sz * rhok_tmp;

            const int off_dm = (i0+ip) + nao*(j0+jp);
            const double dm_ij = dm[off_dm];
            jx += dm_ij * sx;
            jy += dm_ij * sy;
            jz += dm_ij * sz;
        }}
        
        const double rhoj_k = rhoj[kp+k0];
        j3[0] += jx * rhoj_k;
        j3[1] += jy * rhoj_k;
        j3[2] += jz * rhoj_k;
    }
}

__device__
static void write_int3c2e_ip2_jk(JKMatrix jk, double *j3, double* k3, int ksh){
    int *bas_atm = c_bpcache.bas_atm;
    const int atm_id = bas_atm[ksh];
    double *vj = jk.vj;
    double *vk = jk.vk;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ double sdata[THREADSX][THREADSY];

    if (vj != NULL){
        for (int j = 0; j < 3; j++){
            sdata[ty][tx] = j3[j]; __syncthreads();
            if(THREADSX >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSX >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSX >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSX >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (tx == 0) atomicAdd(vj + 3*atm_id+j, sdata[ty][0]);
        }
    }
    if (vk != NULL){
        for (int j = 0; j < 3; j++){
            sdata[ty][tx] = k3[j]; __syncthreads();
            if(THREADSX >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSX >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSX >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSX >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (tx == 0) atomicAdd(vk + 3*atm_id+j, sdata[ty][0]);
        }
    }
}

// Unrolled verion
template <int LI, int LJ, int LK> __global__
void GINTint3c2e_ip2_jk_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
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
    
    constexpr int LK_CEIL = LK + 1;
    constexpr int NROOTS = (LI+LJ+LK_CEIL)/2 + 1;
    constexpr int GSIZE = 3 * NROOTS * (LI+1)*(LJ+1)*(LK_CEIL+1);

    double g[GSIZE];

    const int as_ish = envs.ibase ? ish: jsh; 
    const int as_jsh = envs.ibase ? jsh: ish; 

    double j3[3] = {0.0};
    double k3[3] = {0.0};

    if (active) {
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        for (int kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
            GINTg0_int3c2e<LI, LJ, LK_CEIL>(envs, g, as_ish, as_jsh, ksh, ij, kl);
            const double ak2 = -2.0*c_bpcache.a1[kl];
            GINTkernel_int3c2e_ip2_getjk_direct<LI, LJ, LK>(envs, jk, j3, k3, g, ak2, ish, jsh, ksh);
        }}
    }
    write_int3c2e_ip2_jk(jk, j3, k3, ksh);
}

__device__
static void GINTkernel_int3c2e_ip2_getjk_direct(GINTEnvVars envs, JKMatrix jk, 
        double* j3, double* k3, double *g, const double ak2,
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

            int ix = dk * idx[loc_k] + dj * idx[loc_j] + di * idx[loc_i];
            int iy = dk * idy[loc_k] + dj * idy[loc_j] + di * idy[loc_i] + g_size;
            int iz = dk * idz[loc_k] + dj * idz[loc_j] + di * idz[loc_i] + g_size * 2;

            const int k_idx = idx[loc_k];
            const int k_idy = idy[loc_k];
            const int k_idz = idz[loc_k];

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < nrys_roots; ++ir, ++ix, ++iy, ++iz){
                const double gx = g[ix];
                const double gy = g[iy];
                const double gz = g[iz];

                double fx = ak2*g[ix+dk];
                double fy = ak2*g[iy+dk];
                double fz = ak2*g[iz+dk];

                fx += k_idx>0 ? k_idx*g[ix-dk] : 0.0;
                fy += k_idy>0 ? k_idy*g[iy-dk] : 0.0;
                fz += k_idz>0 ? k_idz*g[iz-dk] : 0.0;

                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }

            const int off_rhok = (ip+i0) + nao*(jp+j0) + (kp+k0)*nao*nao;
            const double rhok_tmp = rhok[off_rhok];
            k3[0] += sx * rhok_tmp;
            k3[1] += sy * rhok_tmp;
            k3[2] += sz * rhok_tmp;
        }
        return;
    }

    if (rhok == NULL){
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

            const int k_idx = idx[loc_k];
            const int k_idy = idy[loc_k];
            const int k_idz = idz[loc_k];

            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
#pragma unroll
            for (int ir = 0; ir < nrys_roots; ++ir){
                const double gx = g[ix+ir];
                const double gy = g[iy+ir];
                const double gz = g[iz+ir];

                double fx = ak2*g[ix+ir+dk];
                double fy = ak2*g[iy+ir+dk];
                double fz = ak2*g[iz+ir+dk];

                fx += k_idx>0 ? k_idx*g[ix+ir-dk] : 0.0;
                fy += k_idy>0 ? k_idy*g[iy+ir-dk] : 0.0;
                fz += k_idz>0 ? k_idz*g[iz+ir-dk] : 0.0;

                sx += fx * gy * gz;
                sy += gx * fy * gz;
                sz += gx * gy * fz;
            }
            const int off_dm = (ip+i0) + nao*(jp+j0);
            const double dm_rhoj = dm[off_dm] * rhoj[kp + k0];
            j3[0] += dm_rhoj * sx;
            j3[1] += dm_rhoj * sy;
            j3[2] += dm_rhoj * sz;
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

        const int k_idx = idx[loc_k];
        const int k_idy = idy[loc_k];
        const int k_idz = idz[loc_k];

        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
#pragma unroll
        for (int ir = 0; ir < nrys_roots; ++ir){
            const double gx = g[ix+ir];
            const double gy = g[iy+ir];
            const double gz = g[iz+ir];

            double fx = ak2*g[ix+ir+dk];
            double fy = ak2*g[iy+ir+dk];
            double fz = ak2*g[iz+ir+dk];

            fx += k_idx>0 ? k_idx*g[ix+ir-dk] : 0.0;
            fy += k_idy>0 ? k_idy*g[iy+ir-dk] : 0.0;
            fz += k_idz>0 ? k_idz*g[iz+ir-dk] : 0.0;

            sx += fx * gy * gz;
            sy += gx * fy * gz;
            sz += gx * gy * fz;
        }

        const int off_rhok = (i0+ip) + nao*(j0+jp) + (k0+kp)*nao*nao;
        const double rhok_tmp = rhok[off_rhok];
        k3[0] += sx * rhok_tmp;
        k3[1] += sy * rhok_tmp;
        k3[2] += sz * rhok_tmp;

        const int off_dm = (i0+ip) + nao*(jp+j0);
        const double dm_rhoj = dm[off_dm] * rhoj[k0+kp];
        j3[0] += dm_rhoj * sx;
        j3[1] += dm_rhoj * sy;
        j3[2] += dm_rhoj * sz;
    }
}


// General version
__global__
void GINTint3c2e_ip2_jk_general_kernel(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
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
        const double ak2 = -2.0* c_bpcache.a1[kl];
        GINTkernel_int3c2e_ip2_getjk_direct(envs, jk, j3, k3, g, ak2, ish, jsh, ksh);
    }}
    
    constexpr int nthreads = THREADSX * THREADSY;
    int *bas_atm = c_bpcache.bas_atm;
    const int atm_id = bas_atm[ksh];
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
static void GINTint3c2e_ip2_jk_kernel001(GINTEnvVars envs, JKMatrix jk, BasisProdOffsets offsets)
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

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;

    prim_ij0 = prim_ij;
    prim_ij1 = prim_ij + nprim_ij;
    prim_kl0 = prim_kl;
    prim_kl1 = prim_kl + nprim_kl;
    for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
        const double ak2 = -2.0*a1[kl];
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
        const double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
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
        const double tmp1 = u2 / (u2 * aijkl + a1);
        const double tmp3 = tmp1 * aij;
        const double c0px = tmp3 * xijxkl;
        const double c0py = tmp3 * yijykl;
        const double c0pz = tmp3 * zijzkl;
        const double g_0 = 1;
        const double g_1 = c0px;
        const double g_2 = 1;
        const double g_3 = c0py;
        const double g_4 = weight0 * fac;
        const double g_5 = c0pz * g_4;

        const double f_1 = ak2 * g_1;
        const double f_3 = ak2 * g_3;
        const double f_5 = ak2 * g_5;

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
    const int atm_id = bas_atm[ksh];
    if (vj != NULL){
        const int off_dm = i0 + nao*j0;
        const double rhoj_tmp = dm[off_dm] * rhoj[k0];
        double vj_tmp[3];
        vj_tmp[0] = gout0 * rhoj_tmp;
        vj_tmp[1] = gout1 * rhoj_tmp;
        vj_tmp[2] = gout2 * rhoj_tmp;
        for (int j = 0; j < 3; j++){
            sdata[ty][tx] = vj_tmp[j]; __syncthreads();
            if(THREADSX >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSX >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSX >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSX >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (tx == 0) atomicAdd(vj + 3*atm_id+j, sdata[ty][0]);
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
            sdata[ty][tx] = vk_tmp[j]; __syncthreads();
            if(THREADSX >= 16 && tx<8) sdata[ty][tx] += sdata[ty][tx+8]; __syncthreads();
            if(THREADSX >= 8  && tx<4) sdata[ty][tx] += sdata[ty][tx+4]; __syncthreads();
            if(THREADSX >= 4  && tx<2) sdata[ty][tx] += sdata[ty][tx+2]; __syncthreads();
            if(THREADSX >= 2  && tx<1) sdata[ty][tx] += sdata[ty][tx+1]; __syncthreads();
            if (tx == 0) atomicAdd(vk + 3*atm_id+j, sdata[ty][0]);
        }
    }
}
