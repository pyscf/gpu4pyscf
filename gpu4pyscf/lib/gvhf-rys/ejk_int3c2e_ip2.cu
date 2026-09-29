/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "build_rys_gxyz.cuh"

#define THREADS         256
#define BLOCK_SIZE      16

__global__ static
void ejk_int3c2e_ipip_kernel(double *ejk, double *dm, double *density_auxvec,
                             double omega, double lr_factor, double sr_factor,
                             RysIntEnvVars envs, int *shl_pair_offsets,
                             uint32_t *bas_ij_idx, int *ksh_offsets, int *gout_stride_lookup,
                             int *ao_pair_loc, int aux_offset, int naux)
{
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ int nao;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 - nbas * ish0;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        nksh = ksh1 - ksh0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        int lij = li + lj + 2;
        nroots = (lij + lk) / 2 + 1;
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[nbas];
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        int stride_j = li + 2;
        int stride_k = stride_j * (lj + 2);
        g_size = stride_k * (lk + 1);
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
        aux_per_block = min(nst_per_block, BLOCK_SIZE);
        nsp_per_block = nst_per_block / aux_per_block;
    }
    __syncthreads();
    register int gout_id = thread_id / nst_per_block;
    register int st_id = thread_id - gout_id * nst_per_block;
    register int sp_id = st_id / aux_per_block;
    register int aux_id = st_id - sp_id * aux_per_block;

    int gx_len = g_size * nst_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 6 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+6) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        double out_ixx = 0; // on the same center
        double out_ixy = 0;
        double out_ixz = 0;
        double out_iyy = 0;
        double out_iyz = 0;
        double out_izz = 0;
        double out_jxx = 0;
        double out_jxy = 0;
        double out_jxz = 0;
        double out_jyy = 0;
        double out_jyz = 0;
        double out_jzz = 0;
        double out_x_x = 0; // on two centers
        double out_x_y = 0;
        double out_x_z = 0;
        double out_y_x = 0;
        double out_y_y = 0;
        double out_y_z = 0;
        double out_z_x = 0;
        double out_z_y = 0;
        double out_z_z = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int i0 = envs.ao_loc[ish];
        int j0 = envs.ao_loc[jsh];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = env[rj+0] - env[ri+0];
        double yjyi = env[rj+1] - env[ri+1];
        double zjzi = env[rj+2] - env[ri+2];
        __syncthreads();
        if (gout_id == 0 && aux_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }

            double v_ixx = 0; // on the same center
            double v_ixy = 0;
            double v_ixz = 0;
            double v_iyy = 0;
            double v_iyz = 0;
            double v_izz = 0;
            double v_jxx = 0;
            double v_jxy = 0;
            double v_jxz = 0;
            double v_jyy = 0;
            double v_jyz = 0;
            double v_jzz = 0;
            double v1xx = 0; // on two centers
            double v1xy = 0;
            double v1xz = 0;
            double v1yx = 0;
            double v1yy = 0;
            double v1yz = 0;
            double v1zx = 0;
            double v1zy = 0;
            double v1zz = 0;

            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                __syncthreads();
                if (gout_id == 0) {
                    double theta_ij = ai * aj_aij;
                    double xjxi = rjri[0*nsp_per_block];
                    double yjyi = rjri[1*nsp_per_block];
                    double zjzi = rjri[2*nsp_per_block];
                    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    double Kab = theta_ij * rr_ij;
                    double fac_ij = PI_FAC;
                    if (ish == jsh) {
                        fac_ij *= .5;
                    } else if (ish < jsh) {
                        fac_ij = 0;
                    }
                    double cicj = fac_ij * env[ci+ip] * env[cj+jp];
                    gx[gx_len] = cicj * exp(-Kab);
                    double xij = xjxi * aj_aij + env[ri+0];
                    double yij = yjyi * aj_aij + env[ri+1];
                    double zij = zjzi * aj_aij + env[ri+2];
                    double xk = env[rk+0];
                    double yk = env[rk+1];
                    double zk = env[rk+2];
                    double xpq = xij - xk;
                    double ypq = yij - yk;
                    double zpq = zij - zk;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = env[expk+kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                    }
                    double xpq = Rpq[0*nst_per_block];
                    double ypq = Rpq[1*nst_per_block];
                    double zpq = Rpq[2*nst_per_block];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    rys_roots_for_k(nroots, theta, rr, rw, omega, lr_factor, sr_factor,
                                    nst_per_block, gout_stride, gout_id);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int lij = li + lj + 2;
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 2);
                        BUILD_3C_GXYZ(lj+1, lk, nsp_per_block, pair_ij < shl_pair1 && kidx < ksh1);
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            int nsp = nsp_per_block;
                            int i_1 =          nst_per_block;
                            int j_1 = stride_j*nst_per_block;
                            int nfi = c_nf[li];
                            int nfj = c_nf[lj];
                            int nfij = nfi * nfj;
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            float div_nfij = div_nfi * div_nfj;
                            double ai2 = ai * 2;
                            double aj2 = aj * 2;
                            for (int n = gout_id; n < nf; n+=gout_stride) {
                                uint32_t k = n * div_nfij;
                                uint32_t ij = n - k * nfij;
                                uint32_t j = ij * div_nfi;
                                uint32_t i = ij - j * nfi;
                                int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                                int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                                int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                                int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                                int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                                int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                                int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
                                int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
                                int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
                                double dm_ijk;
                                if (density_auxvec == NULL) {
                                    dm_ijk = dm_tensor[ij*naux + k*nksh];
                                } else {
                                    dm_ijk = dm_tensor[j*nao+i] * density_auxvec[k0+k];
                                }
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                                int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nst;
                                int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nst;
                                double Ix = gx[addrx];
                                double Iy = gx[addry];
                                double Iz = gx[addrz];
                                double Ix_d = Ix * dm_ijk;
                                double Iy_d = Iy * dm_ijk;
                                double Iz_d = Iz * dm_ijk;
                                double prod_yz = Iy * Iz_d;
                                double prod_xz = Ix * Iz_d;
                                double prod_xy = Ix * Iy_d;
                                double gix = gx[addrx+i_1];
                                double giy = gx[addry+i_1];
                                double giz = gx[addrz+i_1];
                                double gjx = gx[addrx+j_1];
                                double gjy = gx[addry+j_1];
                                double gjz = gx[addrz+j_1];

                                double f3x, f3y, f3z;
                                double _gx_inc2, _gy_inc2, _gz_inc2;
                                double fjx = aj2 * gjx;
                                double fjy = aj2 * gjy;
                                double fjz = aj2 * gjz;
                                if (jx > 0) { fjx -= jx * gx[addrx-j_1]; }
                                if (jy > 0) { fjy -= jy * gx[addry-j_1]; }
                                if (jz > 0) { fjz -= jz * gx[addrz-j_1]; }

                                double fix = ai2 * gix;
                                double fiy = ai2 * giy;
                                double fiz = ai2 * giz;
                                if (ix > 0) { fix -= ix * gx[addrx-i_1]; }
                                if (iy > 0) { fiy -= iy * gx[addry-i_1]; }
                                if (iz > 0) { fiz -= iz * gx[addrz-i_1]; }

                                double gijx = gx[addrx+i_1+j_1];
                                double gijy = gx[addry+i_1+j_1];
                                double gijz = gx[addrz+i_1+j_1];
                                f3x = ai2 * gijx;
                                f3y = ai2 * gijy;
                                f3z = ai2 * gijz;
                                if (ix > 0) { f3x -= ix * gx[addrx-i_1+j_1]; }
                                if (iy > 0) { f3y -= iy * gx[addry-i_1+j_1]; }
                                if (iz > 0) { f3z -= iz * gx[addrz-i_1+j_1]; }
                                f3x *= aj2;
                                f3y *= aj2;
                                f3z *= aj2;
                                if (jx > 0) {
                                    double fx = ai2 * gx[addrx+i_1-j_1];
                                    if (ix > 0) { fx -= ix * gx[addrx-i_1-j_1]; }
                                    f3x -= jx * fx;
                                }
                                if (jy > 0) {
                                    double fy = ai2 * gx[addry+i_1-j_1];
                                    if (iy > 0) { fy -= iy * gx[addry-i_1-j_1]; }
                                    f3y -= jy * fy;
                                }
                                if (jz > 0) {
                                    double fz = ai2 * gx[addrz+i_1-j_1];
                                    if (iz > 0) { fz -= iz * gx[addrz-i_1-j_1]; }
                                    f3z -= jz * fz;
                                }
                                v1xx += f3x * prod_yz;
                                v1yy += f3y * prod_xz;
                                v1zz += f3z * prod_xy;
                                v1xy += fix * fjy * Iz_d;
                                v1xz += fix * fjz * Iy_d;
                                v1yx += fiy * fjx * Iz_d;
                                v1yz += fiy * fjz * Ix_d;
                                v1zx += fiz * fjx * Iy_d;
                                v1zy += fiz * fjy * Ix_d;
                                double xjxi = rjri[0*nsp];
                                double yjyi = rjri[1*nsp];
                                double zjzi = rjri[2*nsp];
                                _gx_inc2 = gijx - gjx * xjxi;
                                _gy_inc2 = gijy - gjy * yjyi;
                                _gz_inc2 = gijz - gjz * zjzi;
                                f3x = aj2 * (aj2 * _gx_inc2 - (2*jx+1) * Ix);
                                f3y = aj2 * (aj2 * _gy_inc2 - (2*jy+1) * Iy);
                                f3z = aj2 * (aj2 * _gz_inc2 - (2*jz+1) * Iz);
                                if (jx > 1) { f3x += jx*(jx-1) * gx[addrx-j_1*2]; }
                                if (jy > 1) { f3y += jy*(jy-1) * gx[addry-j_1*2]; }
                                if (jz > 1) { f3z += jz*(jz-1) * gx[addrz-j_1*2]; }
                                v_jxx += f3x * prod_yz;
                                v_jyy += f3y * prod_xz;
                                v_jzz += f3z * prod_xy;
                                v_jxy += fjx * fjy * Iz_d;
                                v_jxz += fjx * fjz * Iy_d;
                                v_jyz += fjy * fjz * Ix_d;

                                _gx_inc2 = gijx + gix * xjxi;
                                _gy_inc2 = gijy + giy * yjyi;
                                _gz_inc2 = gijz + giz * zjzi;
                                f3x = ai2 * (ai2 * _gx_inc2 - (2*ix+1) * Ix);
                                f3y = ai2 * (ai2 * _gy_inc2 - (2*iy+1) * Iy);
                                f3z = ai2 * (ai2 * _gz_inc2 - (2*iz+1) * Iz);
                                if (ix > 1) { f3x += ix*(ix-1) * gx[addrx-i_1*2]; }
                                if (iy > 1) { f3y += iy*(iy-1) * gx[addry-i_1*2]; }
                                if (iz > 1) { f3z += iz*(iz-1) * gx[addrz-i_1*2]; }
                                v_ixx += f3x * prod_yz;
                                v_iyy += f3y * prod_xz;
                                v_izz += f3z * prod_xy;
                                v_ixy += fix * fiy * Iz_d;
                                v_ixz += fix * fiz * Iy_d;
                                v_iyz += fiy * fiz * Ix_d;
                            }
                        }
                    }
                }
            }
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                out_ixx += v_ixx;
                out_ixy += v_ixy;
                out_ixz += v_ixz;
                out_iyy += v_iyy;
                out_iyz += v_iyz;
                out_izz += v_izz;
                out_jxx += v_jxx;
                out_jxy += v_jxy;
                out_jxz += v_jxz;
                out_jyy += v_jyy;
                out_jyz += v_jyz;
                out_jzz += v_jzz;
                out_x_x += v1xx;
                out_x_y += v1xy;
                out_x_z += v1xz;
                out_y_x += v1yx;
                out_y_y += v1yy;
                out_y_z += v1yz;
                out_z_x += v1zx;
                out_z_y += v1zy;
                out_z_z += v1zz;
                double v_kxx = v_ixx + v_jxx + 2 * v1xx;
                double v_kyy = v_iyy + v_jyy + 2 * v1yy;
                double v_kzz = v_izz + v_jzz + 2 * v1zz;
                double v_kxy = v_ixy + v_jxy + v1xy + v1yx;
                double v_kxz = v_ixz + v_jxz + v1xz + v1zx;
                double v_kyz = v_iyz + v_jyz + v1yz + v1zy;
                double v_ixkx = -v1xx - v_ixx; // = -ixix - ixjx
                double v_iyky = -v1yy - v_iyy;
                double v_izkz = -v1zz - v_izz;
                double v_ixky = -v1xy - v_ixy; // = -ixiy - ixjy
                double v_ixkz = -v1xz - v_ixz;
                double v_iykx = -v1yx - v_ixy;
                double v_iykz = -v1yz - v_iyz;
                double v_izkx = -v1zx - v_ixz;
                double v_izky = -v1zy - v_iyz;
                double v_jxkx = -v1xx - v_jxx; // = -jxix - jxjx
                double v_jyky = -v1yy - v_jyy;
                double v_jzkz = -v1zz - v_jzz;
                double v_jxky = -v1yx - v_jxy; // = -jxiy - jxjy
                double v_jxkz = -v1zx - v_jxz;
                double v_jykx = -v1xy - v_jxy;
                double v_jykz = -v1zy - v_jyz;
                double v_jzkx = -v1xz - v_jxz;
                double v_jzky = -v1yz - v_jyz;
                int ia = bas[ish*BAS_SLOTS+ATOM_OF];
                int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                int natm = envs.natm;
                atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx * .5);
                atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy     );
                atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy * .5);
                atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz     );
                atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz     );
                atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz * .5);

                atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
                atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
                atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
                atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
                atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
                atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
                atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
                atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
                atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
                atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
                atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
                atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
                atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
                atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
                atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
                atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
                atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
                atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
            }
        }
        if (pair_ij < shl_pair1) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            int natm = envs.natm;
            atomicAdd(ejk + (ia*natm+ja)*9 + 0, out_x_x);
            atomicAdd(ejk + (ia*natm+ja)*9 + 1, out_x_y);
            atomicAdd(ejk + (ia*natm+ja)*9 + 2, out_x_z);
            atomicAdd(ejk + (ia*natm+ja)*9 + 3, out_y_x);
            atomicAdd(ejk + (ia*natm+ja)*9 + 4, out_y_y);
            atomicAdd(ejk + (ia*natm+ja)*9 + 5, out_y_z);
            atomicAdd(ejk + (ia*natm+ja)*9 + 6, out_z_x);
            atomicAdd(ejk + (ia*natm+ja)*9 + 7, out_z_y);
            atomicAdd(ejk + (ia*natm+ja)*9 + 8, out_z_z);
            atomicAdd(ejk + (ia*natm+ia)*9 + 0, out_ixx*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 3, out_ixy);
            atomicAdd(ejk + (ia*natm+ia)*9 + 4, out_iyy*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 6, out_ixz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 7, out_iyz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 8, out_izz*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 0, out_jxx*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 3, out_jxy);
            atomicAdd(ejk + (ja*natm+ja)*9 + 4, out_jyy*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 6, out_jxz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 7, out_jyz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 8, out_jzz*.5);
        }
    }
}

extern "C" {
int ejk_int3c2e_ip2(double *ejk, double *dm, double *density_auxvec,
                    double omega, double lr_factor, double sr_factor,
                    RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int aux_offset, int naux)
{
    cudaFuncSetAttribute(ejk_int3c2e_ipip_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    ejk_int3c2e_ipip_kernel<<<blocks, THREADS, shm_size>>>(
            ejk, dm, density_auxvec, omega, lr_factor, sr_factor, *envs,
            shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, aux_offset, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip2: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
