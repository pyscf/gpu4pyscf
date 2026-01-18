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
#include "rys_roots.cu"
#include "rys_contract_k.cuh"

#define THREADS         256
#define BLOCK_SIZE      16

__global__ static
void ejk_int3c2e_ip2_kernel(double *ejk, double *dm, double *density_auxvec,
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
    __shared__ int li, lj, lij, lk, nroots, nf;
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
        lij = li + lj + 2;
        nroots = (lij + lk) / 2 + 1;
        double omega = env[PTR_RANGE_OMEGA];
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
        g_size = stride_k * (lk + 3);
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
    double *rjri = shared_memory;
    double *Rpq = shared_memory + nsp_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 7 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+7) + st_id;
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
            rjri[sp_id+0*nsp_per_block] = xjxi;
            rjri[sp_id+1*nsp_per_block] = yjyi;
            rjri[sp_id+2*nsp_per_block] = zjzi;
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
                    double xjxi = rjri[sp_id+0*nsp_per_block];
                    double yjyi = rjri[sp_id+1*nsp_per_block];
                    double zjzi = rjri[sp_id+2*nsp_per_block];
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
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                    Rpq[3*nst_per_block] = rr;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = env[expk+kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                    }
                    double omega = env[PTR_RANGE_OMEGA];
                    //TODO: rys_roots_for_k
                    rys_roots_rs(nroots, theta, Rpq[3*nst_per_block], omega,
                                 rw, nst_per_block, gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 1);
                        int i_1 =          nst_per_block;
                        int j_1 = stride_j*nst_per_block;
                        int k_1 = stride_k*nst_per_block;
                        int nst = nst_per_block;
                        __syncthreads();
                        if (gout_id == 0) {
                            gx[gx_len*2] = rw[(irys*2+1)*nst];
                        }
                        double rt = rw[ irys*2   *nst];
                        double rt_aa = rt / (aij + ak);
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double s0x, s1x, s2x;
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            double *_gx = gx + n * gx_len;
                            double Rpa = rjri[sp_id+n*nsp_per_block] * aj_aij;
                            //double c0x = Rpa[ir] - rt_aij * Rpq[n];
                            double c0x = Rpa - rt_aij * Rpq[n*nst];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nst] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nst] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                        int lij3 = (lij+1)*3;
                        double rt_ak  = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/ak  * (1 - rt_ak );
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n - i*3; // TODO: remove _ix for nroots > 2
                            double *_gx = gx + (i + _ix * g_size) * nst;
                            double cpx = rt_ak * Rpq[_ix*nst];
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nst];
                                }
                                _gx[stride_k*nst] = s1x;
                            }
                            for (int k = 1; k < lk+2; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nst];
                                    }
                                    _gx[(k*stride_k+stride_k)*nst] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }

                        __syncthreads();
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            int lk3 = (lk+3)*3;
                            for (int m = gout_id; m < lk3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m - k*3;
                                double xjxi = rjri[sp_id+_ix*nsp_per_block];
                                double *_gx = gx + (_ix*g_size + k*stride_k) * nst;
                                for (int j = 0; j <= lj; ++j) {
                                    int ij = (lij-j) + j*stride_j;
                                    s1x = _gx[ij*nst];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[ij*nst];
                                        _gx[(ij+stride_j)*nst] = s1x - xjxi * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                        __syncthreads();
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            int nfi = c_nf[li];
                            int nfj = c_nf[lj];
                            int nfij = nfi * nfj;
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            float div_nfij = div_nfi * div_nfj;
                            double ai2 = ai * 2;
                            double aj2 = aj * 2;
                            double ak2 = ak * 2;
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
                                double gkx = gx[addrx+k_1];
                                double gky = gx[addry+k_1];
                                double gkz = gx[addrz+k_1];

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

                                double xjxi = rjri[sp_id+0*nsp_per_block];
                                double yjyi = rjri[sp_id+1*nsp_per_block];
                                double zjzi = rjri[sp_id+2*nsp_per_block];
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

                                double fkx = ak2 * gkx;
                                double fky = ak2 * gky;
                                double fkz = ak2 * gkz;
                                if (kx > 0) { fkx -= kx * gx[addrx-k_1]; }
                                if (ky > 0) { fky -= ky * gx[addry-k_1]; }
                                if (kz > 0) { fkz -= kz * gx[addrz-k_1]; }

                                f3x = ak2 * (ak2 * gx[addrx+k_1*2] - (2*kx+1) * Ix);
                                f3y = ak2 * (ak2 * gx[addry+k_1*2] - (2*ky+1) * Iy);
                                f3z = ak2 * (ak2 * gx[addrz+k_1*2] - (2*kz+1) * Iz);
                                if (kx > 1) { f3x += kx*(kx-1) * gx[addrx-k_1*2]; }
                                if (ky > 1) { f3y += ky*(ky-1) * gx[addry-k_1*2]; }
                                if (kz > 1) { f3z += kz*(kz-1) * gx[addrz-k_1*2]; }
                                v_kxx += f3x * prod_yz;
                                v_kyy += f3y * prod_xz;
                                v_kzz += f3z * prod_xy;
                                v_kxy += fkx * fky * Iz_d;
                                v_kxz += fkx * fkz * Iy_d;
                                v_kyz += fky * fkz * Ix_d;

                                v_ixky += fix * fky * Iz_d;
                                v_ixkz += fix * fkz * Iy_d;
                                v_iykx += fiy * fkx * Iz_d;
                                v_iykz += fiy * fkz * Ix_d;
                                v_izkx += fiz * fkx * Iy_d;
                                v_izky += fiz * fky * Ix_d;
                                v_jxky += fjx * fky * Iz_d;
                                v_jxkz += fjx * fkz * Iy_d;
                                v_jykx += fjy * fkx * Iz_d;
                                v_jykz += fjy * fkz * Ix_d;
                                v_jzkx += fjz * fkx * Iy_d;
                                v_jzky += fjz * fky * Ix_d;

                                double gikx = gx[addrx+i_1+k_1];
                                double giky = gx[addry+i_1+k_1];
                                double gikz = gx[addrz+i_1+k_1];
                                double fikx = ai2 * gikx;
                                double fiky = ai2 * giky;
                                double fikz = ai2 * gikz;
                                if (ix > 0) { fikx -= ix * gx[addrx-i_1+k_1]; }
                                if (iy > 0) { fiky -= iy * gx[addry-i_1+k_1]; }
                                if (iz > 0) { fikz -= iz * gx[addrz-i_1+k_1]; }
                                fikx *= ak2;
                                fiky *= ak2;
                                fikz *= ak2;

                                double fjkx = aj2 * (gikx - xjxi * gkx);
                                double fjky = aj2 * (giky - yjyi * gky);
                                double fjkz = aj2 * (gikz - zjzi * gkz);
                                if (jx > 0) { fjkx -= jx * gx[addrx-j_1+k_1]; }
                                if (jy > 0) { fjky -= jy * gx[addry-j_1+k_1]; }
                                if (jz > 0) { fjkz -= jz * gx[addrz-j_1+k_1]; }
                                fjkx *= ak2;
                                fjky *= ak2;
                                fjkz *= ak2;

                                if (kx > 0) {
                                    double gixk = gx[addrx+i_1-k_1];
                                    double fx = ai2 * gixk;
                                    if (ix > 0) { fx -= ix * gx[addrx-i_1-k_1]; }
                                    fikx -= kx * fx;
                                    fx = aj2 * (gixk - xjxi * gx[addrx-k_1]);
                                    if (jx > 0) { fx -= jx * gx[addrx-j_1-k_1]; }
                                    fjkx -= kx * fx;
                                }
                                if (ky > 0) {
                                    double giyk = gx[addry+i_1-k_1];
                                    double fy = ai2 * giyk;
                                    if (iy > 0) { fy -= iy * gx[addry-i_1-k_1]; }
                                    fiky -= ky * fy;
                                    fy = aj2 * (giyk - yjyi * gx[addry-k_1]);
                                    if (jy > 0) { fy -= jy * gx[addry-j_1-k_1]; }
                                    fjky -= ky * fy;
                                }
                                if (kz > 0) {
                                    double gizk = gx[addrz+i_1-k_1];
                                    double fz = ai2 * gizk;
                                    if (iz > 0) { fz -= iz * gx[addrz-i_1-k_1]; }
                                    fikz -= kz * fz;
                                    fz = aj2 * (gizk - zjzi * gx[addrz-k_1]);
                                    if (jz > 0) { fz -= jz * gx[addrz-j_1-k_1]; }
                                    fjkz -= kz * fz;
                                }

                                v_ixkx += fikx * prod_yz;
                                v_iyky += fiky * prod_xz;
                                v_izkz += fikz * prod_xy;
                                v_jxkx += fjkx * prod_yz;
                                v_jyky += fjky * prod_xz;
                                v_jzkz += fjkz * prod_xy;
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
                    RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int aux_offset, int naux)
{
    cudaFuncSetAttribute(ejk_int3c2e_ip2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    ejk_int3c2e_ip2_kernel<<<blocks, THREADS, shm_size>>>(
            ejk, dm, density_auxvec, *envs, shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, aux_offset, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip2: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
