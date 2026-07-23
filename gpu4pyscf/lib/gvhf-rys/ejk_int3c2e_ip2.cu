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
void ejk_int3c2e_ip2_kernel(double *ejk, double *dm, double *density_auxvec,
                            RysIntEnvVars envs, double omega, double lr_factor,
                            double sr_factor, int *shl_pair_offsets,
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
        double v_ixx = 0;
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
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
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

            double v_kxx = 0;
            double v_kxy = 0;
            double v_kxz = 0;
            double v_kyy = 0;
            double v_kyz = 0;
            double v_kzz = 0;
            double v_ixkx = 0;
            double v_ixky = 0;
            double v_ixkz = 0;
            double v_iykx = 0;
            double v_iyky = 0;
            double v_iykz = 0;
            double v_izkx = 0;
            double v_izky = 0;
            double v_izkz = 0;
            double v_jxkx = 0;
            double v_jxky = 0;
            double v_jxkz = 0;
            double v_jykx = 0;
            double v_jyky = 0;
            double v_jykz = 0;
            double v_jzkx = 0;
            double v_jzky = 0;
            double v_jzkz = 0;

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
                        BUILD_3C_GXYZ(lj+1, nsp_per_block, pair_ij < shl_pair1 && kidx < ksh1);
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
                                double fkkx, fkky, fkkz;
                                double goutx, gouty, goutz;
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
                                fkkx = f3x * 2;
                                fkky = f3y * 2;
                                fkkz = f3z * 2;
                                goutx = f3x * prod_yz;
                                gouty = f3y * prod_xz;
                                goutz = f3z * prod_xy;
                                v1xx += goutx;
                                v1yy += gouty;
                                v1zz += goutz;
                                v_ixkx -= goutx; // ixjx in ixkx = -ixix - ixjx
                                v_iyky -= gouty;
                                v_izkz -= goutz;
                                v_jxkx -= goutx; // jxix in jxkx = -jxix - jxjx
                                v_jyky -= gouty;
                                v_jzkz -= goutz;
                                double goutxy = fix * fjy * Iz_d;
                                double goutxz = fix * fjz * Iy_d;
                                double goutyx = fiy * fjx * Iz_d;
                                double goutyz = fiy * fjz * Ix_d;
                                double goutzx = fiz * fjx * Iy_d;
                                double goutzy = fiz * fjy * Ix_d;
                                v1xy += goutxy;
                                v1xz += goutxz;
                                v1yx += goutyx;
                                v1yz += goutyz;
                                v1zx += goutzx;
                                v1zy += goutzy;
                                v_ixky -= goutxy; // ixky = -ixiy - ixjy
                                v_ixkz -= goutxz;
                                v_iykx -= goutyx;
                                v_iykz -= goutyz;
                                v_izkx -= goutzx;
                                v_izky -= goutzy;
                                v_jxky -= goutyx; // jxky = -jxiy - jxjy
                                v_jxkz -= goutzx;
                                v_jykx -= goutxy;
                                v_jykz -= goutzy;
                                v_jzkx -= goutxz;
                                v_jzky -= goutyz;

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
                                fkkx += f3x;
                                fkky += f3y;
                                fkkz += f3z;
                                goutx = f3x * prod_yz;
                                gouty = f3y * prod_xz;
                                goutz = f3z * prod_xy;
                                v_jxx += goutx;
                                v_jyy += gouty;
                                v_jzz += goutz;
                                v_jxkx -= goutx; // jxjx in jxkx = -jxix - jxjx
                                v_jyky -= gouty;
                                v_jzkz -= goutz;
                                goutz = fjx * fjy * Iz_d;
                                gouty = fjx * fjz * Iy_d;
                                goutx = fjy * fjz * Ix_d;
                                v_jxy += goutz;
                                v_jxz += gouty;
                                v_jyz += goutx;
                                v_jxky -= goutz; // ixky = -ixiy - ixjy
                                v_jxkz -= gouty;
                                v_jykx -= goutz;
                                v_jykz -= goutx;
                                v_jzkx -= gouty;
                                v_jzky -= goutx;

                                _gx_inc2 = gijx + gix * xjxi;
                                _gy_inc2 = gijy + giy * yjyi;
                                _gz_inc2 = gijz + giz * zjzi;
                                f3x = ai2 * (ai2 * _gx_inc2 - (2*ix+1) * Ix);
                                f3y = ai2 * (ai2 * _gy_inc2 - (2*iy+1) * Iy);
                                f3z = ai2 * (ai2 * _gz_inc2 - (2*iz+1) * Iz);
                                if (ix > 1) { f3x += ix*(ix-1) * gx[addrx-i_1*2]; }
                                if (iy > 1) { f3y += iy*(iy-1) * gx[addry-i_1*2]; }
                                if (iz > 1) { f3z += iz*(iz-1) * gx[addrz-i_1*2]; }
                                fkkx += f3x;
                                fkky += f3y;
                                fkkz += f3z;
                                goutx = f3x * prod_yz;
                                gouty = f3y * prod_xz;
                                goutz = f3z * prod_xy;
                                v_ixx += goutx;
                                v_iyy += gouty;
                                v_izz += goutz;
                                v_ixkx -= goutx; // ixix in ixkx = -ixix - ixjx
                                v_iyky -= gouty;
                                v_izkz -= goutz;
                                goutz = fix * fiy * Iz_d;
                                gouty = fix * fiz * Iy_d;
                                goutx = fiy * fiz * Ix_d;
                                v_ixy += goutz;
                                v_ixz += gouty;
                                v_iyz += goutx;
                                v_ixky -= goutz; // ixky = -ixiy - ixjy
                                v_ixkz -= gouty;
                                v_iykx -= goutz;
                                v_iykz -= goutx;
                                v_izkx -= gouty;
                                v_izky -= goutx;

                                double fkx = -fix - fjx;
                                double fky = -fiy - fjy;
                                double fkz = -fiz - fjz;
                                v_kxx += fkkx * prod_yz;
                                v_kyy += fkky * prod_xz;
                                v_kzz += fkkz * prod_xy;
                                v_kxy += fkx * fky * Iz_d;
                                v_kxz += fkx * fkz * Iy_d;
                                v_kyz += fky * fkz * Ix_d;
                            }
                        }
                    }
                }
            }
            if (pair_ij < shl_pair1 && kidx < ksh1) {
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
            atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
            atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
            atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
            atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
            atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        }
    }
}

extern "C" {
int ejk_int3c2e_ip2(double *ejk, double *dm, double *density_auxvec,
                    RysIntEnvVars *envs, double omega, double lr_factor,
                    double sr_factor, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int aux_offset, int naux)
{
    cudaFuncSetAttribute(ejk_int3c2e_ip2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    ejk_int3c2e_ip2_kernel<<<blocks, THREADS, shm_size>>>(
            ejk, dm, density_auxvec, *envs, omega, lr_factor, sr_factor,
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
