#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#define THREADS         256
#define BLOCK_SIZE      16


__device__ inline
void int3c2e_ip1_000(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int thread_id = item.get_local_id(1);
    int nst_per_block = item.get_local_range(1);
    #else
    int thread_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    #endif
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 1;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    double *rjri = shared_memory + sp_id;
    double *rw = shared_memory + BLOCK_SIZE * 4 + thread_id;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += BLOCK_SIZE) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*BLOCK_SIZE] = xjxi;
            rjri[1*BLOCK_SIZE] = yjyi;
            rjri[2*BLOCK_SIZE] = zjzi;
            rjri[3*BLOCK_SIZE] = rr_ij;
        }
        __syncthreads();
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += BLOCK_SIZE) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            double dm_tensor[1];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int ij = 0; ij < 1; ij++, n++) {
                            dm_tensor[n] = dm_local[ij * naux + k * nksh];
                        }
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int j = 0; j < 1; j++) {
#pragma unroll
                            for (int i = 0; i < 1; i++, n++) {
                                dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                            }
                        }
                    }
                }
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            double fxk, fyk, fzk;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double ak2 = ak * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double xjxi = rjri[0*BLOCK_SIZE];
                double yjyi = rjri[1*BLOCK_SIZE];
                double zjzi = rjri[2*BLOCK_SIZE];
                double rr_ij = rjri[3*BLOCK_SIZE];
                double Kab = theta_ij * rr_ij;
                double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                if (ish == jsh) {
                    cijk *= .5;
                } else if (ish < jsh) {
                    cijk = 0;
                }
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double xk = rk[0];
                double yk = rk[1];
                double zk = rk[2];
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        double xjxi = rjri[0*BLOCK_SIZE];
                        double yjyi = rjri[1*BLOCK_SIZE];
                        double zjzi = rjri[2*BLOCK_SIZE];
                        double wt = rw[(2*irys+1)*nst_per_block];
                        double rt = rw[ 2*irys   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        Ix = 1;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[0];
                        prod_xz = Ix * Iz * dm_tensor[0];
                        prod_yz = Iy * Iz * dm_tensor[0];
                        double rt_aij = rt_aa * ak;
                        double c0x = xjxi * aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double rt_ak = rt_aa * aij;
                        double cpx = xpq*rt_ak;
                        double trr_01x = cpx * 1;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, v_kx * 2);
                    atomicAdd(ejk_aux+ka*3+1, v_ky * 2);
                    atomicAdd(ejk_aux+ka*3+2, v_kz * 2);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        if (pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

__device__ inline
void int3c2e_ip1_100(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int thread_id = item.get_local_id(1);
    int nst_per_block = item.get_local_range(1);
    #else
    int thread_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    #endif
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    double *rjri = shared_memory + sp_id;
    double *rw = shared_memory + BLOCK_SIZE * 4 + thread_id;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += BLOCK_SIZE) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*BLOCK_SIZE] = xjxi;
            rjri[1*BLOCK_SIZE] = yjyi;
            rjri[2*BLOCK_SIZE] = zjzi;
            rjri[3*BLOCK_SIZE] = rr_ij;
        }
        __syncthreads();
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += BLOCK_SIZE) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            double dm_tensor[3];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int ij = 0; ij < 3; ij++, n++) {
                            dm_tensor[n] = dm_local[ij * naux + k * nksh];
                        }
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int j = 0; j < 1; j++) {
#pragma unroll
                            for (int i = 0; i < 3; i++, n++) {
                                dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                            }
                        }
                    }
                }
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            double fxk, fyk, fzk;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double ak2 = ak * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double xjxi = rjri[0*BLOCK_SIZE];
                double yjyi = rjri[1*BLOCK_SIZE];
                double zjzi = rjri[2*BLOCK_SIZE];
                double rr_ij = rjri[3*BLOCK_SIZE];
                double Kab = theta_ij * rr_ij;
                double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                if (ish == jsh) {
                    cijk *= .5;
                } else if (ish < jsh) {
                    cijk = 0;
                }
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double xk = rk[0];
                double yk = rk[1];
                double zk = rk[2];
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        double xjxi = rjri[0*BLOCK_SIZE];
                        double yjyi = rjri[1*BLOCK_SIZE];
                        double zjzi = rjri[2*BLOCK_SIZE];
                        double wt = rw[(2*irys+1)*nst_per_block];
                        double rt = rw[ 2*irys   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double b00 = .5 * rt_aa;
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi * aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[0];
                        prod_xz = Ix * Iz * dm_tensor[0];
                        prod_yz = Iy * Iz * dm_tensor[0];
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_110x = trr_20x - xjxi * trr_10x;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double rt_ak = rt_aa * aij;
                        double cpx = xpq*rt_ak;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[1];
                        prod_xz = Ix * Iz * dm_tensor[1];
                        prod_yz = Iy * Iz * dm_tensor[1];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_110y = trr_20y - yjyi * trr_10y;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_01x = cpx * 1;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_tensor[2];
                        prod_xz = Ix * Iz * dm_tensor[2];
                        prod_yz = Iy * Iz * dm_tensor[2];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_110z = trr_20z - zjzi * trr_10z;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, v_kx * 2);
                    atomicAdd(ejk_aux+ka*3+1, v_ky * 2);
                    atomicAdd(ejk_aux+ka*3+2, v_kz * 2);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        if (pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

__device__ inline
void int3c2e_ip1_110(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int thread_id = item.get_local_id(1);
    int nst_per_block = item.get_local_range(1);
    #else
    int thread_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    #endif
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    double *rjri = shared_memory + sp_id;
    double *rw = shared_memory + BLOCK_SIZE * 4 + thread_id;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += BLOCK_SIZE) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*BLOCK_SIZE] = xjxi;
            rjri[1*BLOCK_SIZE] = yjyi;
            rjri[2*BLOCK_SIZE] = zjzi;
            rjri[3*BLOCK_SIZE] = rr_ij;
        }
        __syncthreads();
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += BLOCK_SIZE) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            double dm_tensor[9];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int ij = 0; ij < 9; ij++, n++) {
                            dm_tensor[n] = dm_local[ij * naux + k * nksh];
                        }
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int j = 0; j < 3; j++) {
#pragma unroll
                            for (int i = 0; i < 3; i++, n++) {
                                dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                            }
                        }
                    }
                }
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            double fxk, fyk, fzk;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double ak2 = ak * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double xjxi = rjri[0*BLOCK_SIZE];
                double yjyi = rjri[1*BLOCK_SIZE];
                double zjzi = rjri[2*BLOCK_SIZE];
                double rr_ij = rjri[3*BLOCK_SIZE];
                double Kab = theta_ij * rr_ij;
                double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                if (ish == jsh) {
                    cijk *= .5;
                } else if (ish < jsh) {
                    cijk = 0;
                }
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double xk = rk[0];
                double yk = rk[1];
                double zk = rk[2];
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        double xjxi = rjri[0*BLOCK_SIZE];
                        double yjyi = rjri[1*BLOCK_SIZE];
                        double zjzi = rjri[2*BLOCK_SIZE];
                        double wt = rw[(2*irys+1)*nst_per_block];
                        double rt = rw[ 2*irys   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double b00 = .5 * rt_aa;
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi * aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double hrr_110x = trr_20x - xjxi * trr_10x;
                        Ix = hrr_110x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[0];
                        prod_xz = Ix * Iz * dm_tensor[0];
                        prod_yz = Iy * Iz * dm_tensor[0];
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_210x = trr_30x - xjxi * trr_20x;
                        fxi = ai2 * hrr_210x;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxi -= 1 * hrr_010x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_120x = hrr_210x - xjxi * hrr_110x;
                        fxj = aj2 * hrr_120x;
                        fxj -= 1 * trr_10x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double rt_ak = rt_aa * aij;
                        double cpx = xpq*rt_ak;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double hrr_111x = trr_21x - xjxi * trr_11x;
                        fxk = ak2 * hrr_111x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = hrr_010x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[1];
                        prod_xz = Ix * Iz * dm_tensor[1];
                        prod_yz = Iy * Iz * dm_tensor[1];
                        fxi = ai2 * hrr_110x;
                        v_ix += fxi * prod_yz;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_020x = hrr_110x - xjxi * hrr_010x;
                        fxj = aj2 * hrr_020x;
                        fxj -= 1 * 1;
                        v_jx += fxj * prod_yz;
                        double hrr_110y = trr_20y - yjyi * trr_10y;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_01x = cpx * 1;
                        double hrr_011x = trr_11x - xjxi * trr_01x;
                        fxk = ak2 * hrr_011x;
                        v_kx += fxk * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = hrr_010x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_tensor[2];
                        prod_xz = Ix * Iz * dm_tensor[2];
                        prod_yz = Iy * Iz * dm_tensor[2];
                        fxi = ai2 * hrr_110x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_020x;
                        fxj -= 1 * 1;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_110z = trr_20z - zjzi * trr_10z;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * hrr_011x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_10x;
                        Iy = hrr_010y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[3];
                        prod_xz = Ix * Iz * dm_tensor[3];
                        prod_yz = Iy * Iz * dm_tensor[3];
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * hrr_110y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        double hrr_020y = hrr_110y - yjyi * hrr_010y;
                        fyj = aj2 * hrr_020y;
                        fyj -= 1 * fac1;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double hrr_011y = trr_11y - yjyi * trr_01y;
                        fyk = ak2 * hrr_011y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = hrr_110y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[4];
                        prod_xz = Ix * Iz * dm_tensor[4];
                        prod_yz = Iy * Iz * dm_tensor[4];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_210y = trr_30y - yjyi * trr_20y;
                        fyi = ai2 * hrr_210y;
                        fyi -= 1 * hrr_010y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_120y = hrr_210y - yjyi * hrr_110y;
                        fyj = aj2 * hrr_120y;
                        fyj -= 1 * trr_10y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_111y = trr_21y - yjyi * trr_11y;
                        fyk = ak2 * hrr_111y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = hrr_010y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_tensor[5];
                        prod_xz = Ix * Iz * dm_tensor[5];
                        prod_yz = Iy * Iz * dm_tensor[5];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * hrr_110y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_020y;
                        fyj -= 1 * fac1;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * hrr_011y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = hrr_010z;
                        prod_xy = Ix * Iy * dm_tensor[6];
                        prod_xz = Ix * Iz * dm_tensor[6];
                        prod_yz = Iy * Iz * dm_tensor[6];
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * hrr_110z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_020z = hrr_110z - zjzi * hrr_010z;
                        fzj = aj2 * hrr_020z;
                        fzj -= 1 * wt;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double hrr_011z = trr_11z - zjzi * trr_01z;
                        fzk = ak2 * hrr_011z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = hrr_010z;
                        prod_xy = Ix * Iy * dm_tensor[7];
                        prod_xz = Ix * Iz * dm_tensor[7];
                        prod_yz = Iy * Iz * dm_tensor[7];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * hrr_110z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_020z;
                        fzj -= 1 * wt;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * hrr_011z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = fac1;
                        Iz = hrr_110z;
                        prod_xy = Ix * Iy * dm_tensor[8];
                        prod_xz = Ix * Iz * dm_tensor[8];
                        prod_yz = Iy * Iz * dm_tensor[8];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_210z = trr_30z - zjzi * trr_20z;
                        fzi = ai2 * hrr_210z;
                        fzi -= 1 * hrr_010z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_120z = hrr_210z - zjzi * hrr_110z;
                        fzj = aj2 * hrr_120z;
                        fzj -= 1 * trr_10z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_111z = trr_21z - zjzi * trr_11z;
                        fzk = ak2 * hrr_111z;
                        v_kz += fzk * prod_xy;
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, v_kx * 2);
                    atomicAdd(ejk_aux+ka*3+1, v_ky * 2);
                    atomicAdd(ejk_aux+ka*3+2, v_kz * 2);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        if (pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

__device__ inline
void int3c2e_ip1_200(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int thread_id = item.get_local_id(1);
    int nst_per_block = item.get_local_range(1);
    #else
    int thread_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    #endif
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    double *rjri = shared_memory + sp_id;
    double *rw = shared_memory + BLOCK_SIZE * 4 + thread_id;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += BLOCK_SIZE) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*BLOCK_SIZE] = xjxi;
            rjri[1*BLOCK_SIZE] = yjyi;
            rjri[2*BLOCK_SIZE] = zjzi;
            rjri[3*BLOCK_SIZE] = rr_ij;
        }
        __syncthreads();
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += BLOCK_SIZE) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            double dm_tensor[6];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int ij = 0; ij < 6; ij++, n++) {
                            dm_tensor[n] = dm_local[ij * naux + k * nksh];
                        }
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0, k = 0; k < 1; k++) {
#pragma unroll
                        for (int j = 0; j < 1; j++) {
#pragma unroll
                            for (int i = 0; i < 6; i++, n++) {
                                dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                            }
                        }
                    }
                }
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            double fxk, fyk, fzk;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double ak2 = ak * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double xjxi = rjri[0*BLOCK_SIZE];
                double yjyi = rjri[1*BLOCK_SIZE];
                double zjzi = rjri[2*BLOCK_SIZE];
                double rr_ij = rjri[3*BLOCK_SIZE];
                double Kab = theta_ij * rr_ij;
                double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                if (ish == jsh) {
                    cijk *= .5;
                } else if (ish < jsh) {
                    cijk = 0;
                }
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double xk = rk[0];
                double yk = rk[1];
                double zk = rk[2];
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        double xjxi = rjri[0*BLOCK_SIZE];
                        double yjyi = rjri[1*BLOCK_SIZE];
                        double zjzi = rjri[2*BLOCK_SIZE];
                        double wt = rw[(2*irys+1)*nst_per_block];
                        double rt = rw[ 2*irys   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double b00 = .5 * rt_aa;
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi * aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        Ix = trr_20x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[0];
                        prod_xz = Ix * Iz * dm_tensor[0];
                        prod_yz = Iy * Iz * dm_tensor[0];
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        fxi = ai2 * trr_30x;
                        fxi -= 2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_210x = trr_30x - xjxi * trr_20x;
                        fxj = aj2 * hrr_210x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double rt_ak = rt_aa * aij;
                        double cpx = xpq*rt_ak;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        fxk = ak2 * trr_21x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_10x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[1];
                        prod_xz = Ix * Iz * dm_tensor[1];
                        prod_yz = Iy * Iz * dm_tensor[1];
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_110x = trr_20x - xjxi * trr_10x;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        double hrr_110y = trr_20y - yjyi * trr_10y;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_tensor[2];
                        prod_xz = Ix * Iz * dm_tensor[2];
                        prod_yz = Iy * Iz * dm_tensor[2];
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_110z = trr_20z - zjzi * trr_10z;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_20y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[3];
                        prod_xz = Ix * Iz * dm_tensor[3];
                        prod_yz = Iy * Iz * dm_tensor[3];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        fyi = ai2 * trr_30y;
                        fyi -= 2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_210y = trr_30y - yjyi * trr_20y;
                        fyj = aj2 * hrr_210y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_01x = cpx * 1;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        fyk = ak2 * trr_21y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_tensor[4];
                        prod_xz = Ix * Iz * dm_tensor[4];
                        prod_yz = Iy * Iz * dm_tensor[4];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_20z;
                        prod_xy = Ix * Iy * dm_tensor[5];
                        prod_xz = Ix * Iz * dm_tensor[5];
                        prod_yz = Iy * Iz * dm_tensor[5];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        fzi = ai2 * trr_30z;
                        fzi -= 2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_210z = trr_30z - zjzi * trr_20z;
                        fzj = aj2 * hrr_210z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        fzk = ak2 * trr_21z;
                        v_kz += fzk * prod_xy;
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, v_kx * 2);
                    atomicAdd(ejk_aux+ka*3+1, v_ky * 2);
                    atomicAdd(ejk_aux+ka*3+2, v_kz * 2);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        if (pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

__device__ inline
void int3c2e_ip1_001(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int thread_id = item.get_local_id(1);
    int nst_per_block = item.get_local_range(1);
    #else
    int thread_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    #endif
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    double *rjri = shared_memory + sp_id;
    double *rw = shared_memory + BLOCK_SIZE * 4 + thread_id;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += BLOCK_SIZE) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*BLOCK_SIZE] = xjxi;
            rjri[1*BLOCK_SIZE] = yjyi;
            rjri[2*BLOCK_SIZE] = zjzi;
            rjri[3*BLOCK_SIZE] = rr_ij;
        }
        __syncthreads();
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += BLOCK_SIZE) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            double dm_tensor[3];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0, k = 0; k < 3; k++) {
#pragma unroll
                        for (int ij = 0; ij < 1; ij++, n++) {
                            dm_tensor[n] = dm_local[ij * naux + k * nksh];
                        }
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0, k = 0; k < 3; k++) {
#pragma unroll
                        for (int j = 0; j < 1; j++) {
#pragma unroll
                            for (int i = 0; i < 1; i++, n++) {
                                dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                            }
                        }
                    }
                }
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            double fxk, fyk, fzk;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double ak2 = ak * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double xjxi = rjri[0*BLOCK_SIZE];
                double yjyi = rjri[1*BLOCK_SIZE];
                double zjzi = rjri[2*BLOCK_SIZE];
                double rr_ij = rjri[3*BLOCK_SIZE];
                double Kab = theta_ij * rr_ij;
                double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                if (ish == jsh) {
                    cijk *= .5;
                } else if (ish < jsh) {
                    cijk = 0;
                }
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double xk = rk[0];
                double yk = rk[1];
                double zk = rk[2];
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        double xjxi = rjri[0*BLOCK_SIZE];
                        double yjyi = rjri[1*BLOCK_SIZE];
                        double zjzi = rjri[2*BLOCK_SIZE];
                        double wt = rw[(2*irys+1)*nst_per_block];
                        double rt = rw[ 2*irys   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double b00 = .5 * rt_aa;
                        double rt_ak = rt_aa * aij;
                        double b01 = .5/ak * (1 - rt_ak);
                        double cpx = xpq*rt_ak;
                        double trr_01x = cpx * 1;
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[0];
                        prod_xz = Ix * Iz * dm_tensor[0];
                        prod_yz = Iy * Iz * dm_tensor[0];
                        double rt_aij = rt_aa * ak;
                        double c0x = xjxi * aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_011x = trr_11x - xjxi * trr_01x;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[1];
                        prod_xz = Ix * Iz * dm_tensor[1];
                        prod_yz = Iy * Iz * dm_tensor[1];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_011y = trr_11y - yjyi * trr_01y;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_tensor[2];
                        prod_xz = Ix * Iz * dm_tensor[2];
                        prod_yz = Iy * Iz * dm_tensor[2];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_011z = trr_11z - zjzi * trr_01z;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, v_kx * 2);
                    atomicAdd(ejk_aux+ka*3+1, v_ky * 2);
                    atomicAdd(ejk_aux+ka*3+2, v_kz * 2);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        if (pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

__device__ inline
void int3c2e_ip1_101(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int thread_id = item.get_local_id(1);
    int nst_per_block = item.get_local_range(1);
    #else
    int thread_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    #endif
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    double *rjri = shared_memory + sp_id;
    double *rw = shared_memory + BLOCK_SIZE * 4 + thread_id;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += BLOCK_SIZE) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*BLOCK_SIZE] = xjxi;
            rjri[1*BLOCK_SIZE] = yjyi;
            rjri[2*BLOCK_SIZE] = zjzi;
            rjri[3*BLOCK_SIZE] = rr_ij;
        }
        __syncthreads();
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += BLOCK_SIZE) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            double dm_tensor[9];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0, k = 0; k < 3; k++) {
#pragma unroll
                        for (int ij = 0; ij < 3; ij++, n++) {
                            dm_tensor[n] = dm_local[ij * naux + k * nksh];
                        }
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0, k = 0; k < 3; k++) {
#pragma unroll
                        for (int j = 0; j < 1; j++) {
#pragma unroll
                            for (int i = 0; i < 3; i++, n++) {
                                dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                            }
                        }
                    }
                }
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            double fxk, fyk, fzk;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double ak2 = ak * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double xjxi = rjri[0*BLOCK_SIZE];
                double yjyi = rjri[1*BLOCK_SIZE];
                double zjzi = rjri[2*BLOCK_SIZE];
                double rr_ij = rjri[3*BLOCK_SIZE];
                double Kab = theta_ij * rr_ij;
                double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                if (ish == jsh) {
                    cijk *= .5;
                } else if (ish < jsh) {
                    cijk = 0;
                }
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double xk = rk[0];
                double yk = rk[1];
                double zk = rk[2];
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        double xjxi = rjri[0*BLOCK_SIZE];
                        double yjyi = rjri[1*BLOCK_SIZE];
                        double zjzi = rjri[2*BLOCK_SIZE];
                        double wt = rw[(2*irys+1)*nst_per_block];
                        double rt = rw[ 2*irys   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double b00 = .5 * rt_aa;
                        double rt_ak = rt_aa * aij;
                        double b01 = .5/ak * (1 - rt_ak);
                        double cpx = xpq*rt_ak;
                        double rt_aij = rt_aa * ak;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi * aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        Ix = trr_11x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[0];
                        prod_xz = Ix * Iz * dm_tensor[0];
                        prod_yz = Iy * Iz * dm_tensor[0];
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        fxi = ai2 * trr_21x;
                        double trr_01x = cpx * 1;
                        fxi -= 1 * trr_01x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_111x = trr_21x - xjxi * trr_11x;
                        fxj = aj2 * hrr_111x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        fxk = ak2 * trr_12x;
                        fxk -= 1 * trr_10x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_01x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[1];
                        prod_xz = Ix * Iz * dm_tensor[1];
                        prod_yz = Iy * Iz * dm_tensor[1];
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_011x = trr_11x - xjxi * trr_01x;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        double hrr_110y = trr_20y - yjyi * trr_10y;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_tensor[2];
                        prod_xz = Ix * Iz * dm_tensor[2];
                        prod_yz = Iy * Iz * dm_tensor[2];
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_110z = trr_20z - zjzi * trr_10z;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_10x;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[3];
                        prod_xz = Ix * Iz * dm_tensor[3];
                        prod_yz = Iy * Iz * dm_tensor[3];
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_110x = trr_20x - xjxi * trr_10x;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        double hrr_011y = trr_11y - yjyi * trr_01y;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_11y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[4];
                        prod_xz = Ix * Iz * dm_tensor[4];
                        prod_yz = Iy * Iz * dm_tensor[4];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        fyi = ai2 * trr_21y;
                        fyi -= 1 * trr_01y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_111y = trr_21y - yjyi * trr_11y;
                        fyj = aj2 * hrr_111y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        fyk = ak2 * trr_12y;
                        fyk -= 1 * trr_10y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_tensor[5];
                        prod_xz = Ix * Iz * dm_tensor[5];
                        prod_yz = Iy * Iz * dm_tensor[5];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_tensor[6];
                        prod_xz = Ix * Iz * dm_tensor[6];
                        prod_yz = Iy * Iz * dm_tensor[6];
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_011z = trr_11z - zjzi * trr_01z;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_tensor[7];
                        prod_xz = Ix * Iz * dm_tensor[7];
                        prod_yz = Iy * Iz * dm_tensor[7];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_11z;
                        prod_xy = Ix * Iy * dm_tensor[8];
                        prod_xz = Ix * Iz * dm_tensor[8];
                        prod_yz = Iy * Iz * dm_tensor[8];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        fzi = ai2 * trr_21z;
                        fzi -= 1 * trr_01z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_111z = trr_21z - zjzi * trr_11z;
                        fzj = aj2 * hrr_111z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        fzk = ak2 * trr_12z;
                        fzk -= 1 * trr_10z;
                        v_kz += fzk * prod_xy;
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, v_kx * 2);
                    atomicAdd(ejk_aux+ka*3+1, v_ky * 2);
                    atomicAdd(ejk_aux+ka*3+2, v_kz * 2);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        if (pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

__device__ inline
void int3c2e_ip1_002(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    #ifdef USE_SYCL
    auto item = syclex::this_work_item::get_nd_item<2>();
    int thread_id = item.get_local_id(1);
    int nst_per_block = item.get_local_range(1);
    #else
    int thread_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    #endif
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    double omega = env[PTR_RANGE_OMEGA];
    if (omega < 0) {
        nroots *= 2;
    }
    double *rjri = shared_memory + sp_id;
    double *rw = shared_memory + BLOCK_SIZE * 4 + thread_id;
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += BLOCK_SIZE) {
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*BLOCK_SIZE] = xjxi;
            rjri[1*BLOCK_SIZE] = yjyi;
            rjri[2*BLOCK_SIZE] = zjzi;
            rjri[3*BLOCK_SIZE] = rr_ij;
        }
        __syncthreads();
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += BLOCK_SIZE) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            double dm_tensor[6];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    double *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0, k = 0; k < 6; k++) {
#pragma unroll
                        for (int ij = 0; ij < 1; ij++, n++) {
                            dm_tensor[n] = dm_local[ij * naux + k * nksh];
                        }
                    }
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                    for (int n = 0, k = 0; k < 6; k++) {
#pragma unroll
                        for (int j = 0; j < 1; j++) {
#pragma unroll
                            for (int i = 0; i < 1; i++, n++) {
                                dm_tensor[n] = dm_local[j*nao+i] * density_auxvec[k0+k];
                            }
                        }
                    }
                }
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            double fxk, fyk, fzk;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
                double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = expi[ip];
                double aj = expj[jp];
                double ak = expk[kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double ak2 = ak * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double xjxi = rjri[0*BLOCK_SIZE];
                double yjyi = rjri[1*BLOCK_SIZE];
                double zjzi = rjri[2*BLOCK_SIZE];
                double rr_ij = rjri[3*BLOCK_SIZE];
                double Kab = theta_ij * rr_ij;
                double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                if (ish == jsh) {
                    cijk *= .5;
                } else if (ish < jsh) {
                    cijk = 0;
                }
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + ri[0];
                double yij = yjyi * aj_aij + ri[1];
                double zij = zjzi * aj_aij + ri[2];
                double xk = rk[0];
                double yk = rk[1];
                double zk = rk[2];
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        double xjxi = rjri[0*BLOCK_SIZE];
                        double yjyi = rjri[1*BLOCK_SIZE];
                        double zjzi = rjri[2*BLOCK_SIZE];
                        double wt = rw[(2*irys+1)*nst_per_block];
                        double rt = rw[ 2*irys   *nst_per_block];
                        double rt_aa = rt / (aij + ak);
                        double b00 = .5 * rt_aa;
                        double rt_ak = rt_aa * aij;
                        double b01 = .5/ak * (1 - rt_ak);
                        double cpx = xpq*rt_ak;
                        double trr_01x = cpx * 1;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        Ix = trr_02x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[0];
                        prod_xz = Ix * Iz * dm_tensor[0];
                        prod_yz = Iy * Iz * dm_tensor[0];
                        double rt_aij = rt_aa * ak;
                        double c0x = xjxi * aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        fxi = ai2 * trr_12x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_012x = trr_12x - xjxi * trr_02x;
                        fxj = aj2 * hrr_012x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        fxk = ak2 * trr_03x;
                        fxk -= 2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_01x;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[1];
                        prod_xz = Ix * Iz * dm_tensor[1];
                        prod_yz = Iy * Iz * dm_tensor[1];
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_011x = trr_11x - xjxi * trr_01x;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        double hrr_011y = trr_11y - yjyi * trr_01y;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_tensor[2];
                        prod_xz = Ix * Iz * dm_tensor[2];
                        prod_yz = Iy * Iz * dm_tensor[2];
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_011z = trr_11z - zjzi * trr_01z;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_02y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_tensor[3];
                        prod_xz = Ix * Iz * dm_tensor[3];
                        prod_yz = Iy * Iz * dm_tensor[3];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        fyi = ai2 * trr_12y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_012y = trr_12y - yjyi * trr_02y;
                        fyj = aj2 * hrr_012y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        fyk = ak2 * trr_03y;
                        fyk -= 2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_tensor[4];
                        prod_xz = Ix * Iz * dm_tensor[4];
                        prod_yz = Iy * Iz * dm_tensor[4];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_02z;
                        prod_xy = Ix * Iy * dm_tensor[5];
                        prod_xz = Ix * Iz * dm_tensor[5];
                        prod_yz = Iy * Iz * dm_tensor[5];
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        fzi = ai2 * trr_12z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_012z = trr_12z - zjzi * trr_02z;
                        fzj = aj2 * hrr_012z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        fzk = ak2 * trr_03z;
                        fzk -= 2 * trr_01z;
                        v_kz += fzk * prod_xy;
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, v_kx * 2);
                    atomicAdd(ejk_aux+ka*3+1, v_ky * 2);
                    atomicAdd(ejk_aux+ka*3+2, v_kz * 2);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        if (pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, v_ix * 2);
            atomicAdd(ejk+ia*3+1, v_iy * 2);
            atomicAdd(ejk+ia*3+2, v_iz * 2);
            atomicAdd(ejk+ja*3+0, v_jx * 2);
            atomicAdd(ejk+ja*3+1, v_jy * 2);
            atomicAdd(ejk+ja*3+2, v_jz * 2);
        }
    }
}

__device__ inline
int int3c2e_ip1_unrolled(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1, int ksh0, int ksh1,
                    int iprim, int jprim, int kprim, int li, int lj, int lk,
                    uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao, double* shared_memory)
{
    int kij_type = lk*25 + li*5 + lj;
    switch (kij_type) {
    case 0: // li=0 lj=0 lk=0
        int3c2e_ip1_000(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            bas_ij_idx, ao_pair_loc, aux_offset, naux, nao, shared_memory); break;
    case 5: // li=1 lj=0 lk=0
        int3c2e_ip1_100(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            bas_ij_idx, ao_pair_loc, aux_offset, naux, nao, shared_memory); break;
    case 6: // li=1 lj=1 lk=0
        int3c2e_ip1_110(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            bas_ij_idx, ao_pair_loc, aux_offset, naux, nao, shared_memory); break;
    case 10: // li=2 lj=0 lk=0
        int3c2e_ip1_200(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            bas_ij_idx, ao_pair_loc, aux_offset, naux, nao, shared_memory); break;
    case 25: // li=0 lj=0 lk=1
        int3c2e_ip1_001(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            bas_ij_idx, ao_pair_loc, aux_offset, naux, nao, shared_memory); break;
    case 30: // li=1 lj=0 lk=1
        int3c2e_ip1_101(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            bas_ij_idx, ao_pair_loc, aux_offset, naux, nao, shared_memory); break;
    case 50: // li=0 lj=0 lk=2
        int3c2e_ip1_002(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            bas_ij_idx, ao_pair_loc, aux_offset, naux, nao, shared_memory); break;
    default: return 0;
    }
    return 1;
}
