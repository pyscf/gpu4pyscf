#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#define THREADS         256
#define BLOCK_SIZE      16


__device__ inline
void int3c2e_ip1_000(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 1;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = hrr_110x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = hrr_010x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = hrr_010x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[3*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = hrr_010y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[4*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = hrr_110y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[5*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = hrr_010y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[6*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = hrr_010z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[7*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = hrr_010z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[8*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = hrr_110z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_20x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[3*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = trr_20y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[4*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[5*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_20z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
void int3c2e_ip1_210(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_210x = trr_30x - xjxi * trr_20x;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = hrr_210x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double hrr_310x = trr_40x - xjxi * trr_30x;
                        fxi = ai2 * hrr_310x;
                        double hrr_110x = trr_20x - xjxi * trr_10x;
                        fxi -= 2 * hrr_110x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_220x = hrr_310x - xjxi * hrr_210x;
                        fxj = aj2 * hrr_220x;
                        fxj -= 1 * trr_20x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double rt_ak = rt_aa * aij;
                        double cpx = xpq*rt_ak;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_211x = trr_31x - xjxi * trr_21x;
                        fxk = ak2 * hrr_211x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = hrr_110x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * hrr_210x;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxi -= 1 * hrr_010x;
                        v_ix += fxi * prod_yz;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_120x = hrr_210x - xjxi * hrr_110x;
                        fxj = aj2 * hrr_120x;
                        fxj -= 1 * trr_10x;
                        v_jx += fxj * prod_yz;
                        double hrr_110y = trr_20y - yjyi * trr_10y;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double hrr_111x = trr_21x - xjxi * trr_11x;
                        fxk = ak2 * hrr_111x;
                        v_kx += fxk * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = hrr_110x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * hrr_210x;
                        fxi -= 1 * hrr_010x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_120x;
                        fxj -= 1 * trr_10x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_110z = trr_20z - zjzi * trr_10z;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * hrr_111x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[3*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+0];
                        }
                        Ix = hrr_010x;
                        Iy = trr_20y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * hrr_110x;
                        v_ix += fxi * prod_yz;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        fyi = ai2 * trr_30y;
                        fyi -= 2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_020x = hrr_110x - xjxi * hrr_010x;
                        fxj = aj2 * hrr_020x;
                        fxj -= 1 * 1;
                        v_jx += fxj * prod_yz;
                        double hrr_210y = trr_30y - yjyi * trr_20y;
                        fyj = aj2 * hrr_210y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_01x = cpx * 1;
                        double hrr_011x = trr_11x - xjxi * trr_01x;
                        fxk = ak2 * hrr_011x;
                        v_kx += fxk * prod_yz;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        fyk = ak2 * trr_21y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[4*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+0];
                        }
                        Ix = hrr_010x;
                        Iy = trr_10y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * hrr_110x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_020x;
                        fxj -= 1 * 1;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * hrr_011x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[5*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+0];
                        }
                        Ix = hrr_010x;
                        Iy = fac1;
                        Iz = trr_20z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * hrr_110x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        fzi = ai2 * trr_30z;
                        fzi -= 2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_020x;
                        fxj -= 1 * 1;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_210z = trr_30z - zjzi * trr_20z;
                        fzj = aj2 * hrr_210z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * hrr_011x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        fzk = ak2 * trr_21z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[6*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_20x;
                        Iy = hrr_010y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_30x;
                        fxi -= 2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * hrr_110y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_210x;
                        v_jx += fxj * prod_yz;
                        double hrr_020y = hrr_110y - yjyi * hrr_010y;
                        fyj = aj2 * hrr_020y;
                        fyj -= 1 * fac1;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_21x;
                        v_kx += fxk * prod_yz;
                        double hrr_011y = trr_11y - yjyi * trr_01y;
                        fyk = ak2 * hrr_011y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[7*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = hrr_110y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * hrr_210y;
                        fyi -= 1 * hrr_010y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        double hrr_120y = hrr_210y - yjyi * hrr_110y;
                        fyj = aj2 * hrr_120y;
                        fyj -= 1 * trr_10y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double hrr_111y = trr_21y - yjyi * trr_11y;
                        fyk = ak2 * hrr_111y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[8*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = hrr_010y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * hrr_110y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_020y;
                        fyj -= 1 * fac1;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * hrr_011y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[9*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+3] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = hrr_210y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_310y = trr_40y - yjyi * trr_30y;
                        fyi = ai2 * hrr_310y;
                        fyi -= 2 * hrr_110y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_220y = hrr_310y - yjyi * hrr_210y;
                        fyj = aj2 * hrr_220y;
                        fyj -= 1 * trr_20y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_211y = trr_31y - yjyi * trr_21y;
                        fyk = ak2 * hrr_211y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[10*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+4] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = hrr_110y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * hrr_210y;
                        fyi -= 1 * hrr_010y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_120y;
                        fyj -= 1 * trr_10y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * hrr_111y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[11*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[1*nao+5] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = hrr_010y;
                        Iz = trr_20z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * hrr_110y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_30z;
                        fzi -= 2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_020y;
                        fyj -= 1 * fac1;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_210z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * hrr_011y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_21z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[12*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_20x;
                        Iy = fac1;
                        Iz = hrr_010z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_30x;
                        fxi -= 2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * hrr_110z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_210x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_020z = hrr_110z - zjzi * hrr_010z;
                        fzj = aj2 * hrr_020z;
                        fzj -= 1 * wt;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_21x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double hrr_011z = trr_11z - zjzi * trr_01z;
                        fzk = ak2 * hrr_011z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[13*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = trr_10y;
                        Iz = hrr_010z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * hrr_110z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_020z;
                        fzj -= 1 * wt;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * hrr_011z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[14*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = hrr_110z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * hrr_210z;
                        fzi -= 1 * hrr_010z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_120z = hrr_210z - zjzi * hrr_110z;
                        fzj = aj2 * hrr_120z;
                        fzj -= 1 * trr_10z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double hrr_111z = trr_21z - zjzi * trr_11z;
                        fzk = ak2 * hrr_111z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[15*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+3] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = trr_20y;
                        Iz = hrr_010z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_30y;
                        fyi -= 2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * hrr_110z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_210y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_020z;
                        fzj -= 1 * wt;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_21y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * hrr_011z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[16*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+4] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = hrr_110z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * hrr_210z;
                        fzi -= 1 * hrr_010z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_120z;
                        fzj -= 1 * trr_10z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * hrr_111z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[17*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[2*nao+5] * density_auxvec[k0+0];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = hrr_210z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_310z = trr_40z - zjzi * trr_30z;
                        fzi = ai2 * hrr_310z;
                        fzi -= 2 * hrr_110z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_220z = hrr_310z - zjzi * hrr_210z;
                        fzj = aj2 * hrr_220z;
                        fzj -= 1 * trr_20z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_211z = trr_31z - zjzi * trr_21z;
                        fzk = ak2 * hrr_211z;
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
void int3c2e_ip1_220(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    constexpr int aux_per_block = 16;
    constexpr int nsp_per_block = 4;
    constexpr int nst_per_block = 64;
    int gout_id = thread_id / 64;
    int st_id = thread_id % 64;
    int sp_id = st_id / aux_per_block;
    int aux_id = st_id - sp_id * aux_per_block;
    extern __shared__ double shared_memory[];
    double *Rpq = shared_memory + st_id;
    double *gx = shared_memory + 192 + st_id;
    double *rw = shared_memory + 4800 + st_id;
    double *rjri = shared_memory + 4800 + nroots * 128 + sp_id;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
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
        if (gout_id == 0 && aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
            rjri[3*nsp_per_block] = rr_ij;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double s0, s1, s2;
            double dm_ijk;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
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
                __syncthreads();
                double xij = rjri[0*nsp_per_block] * aj_aij + ri[0];
                double yij = rjri[1*nsp_per_block] * aj_aij + ri[1];
                double zij = rjri[2*nsp_per_block] * aj_aij + ri[2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                if (gout_id == 0) {
                    double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                    if (ish == jsh) {
                        cijk *= .5;
                    } else if (ish < jsh) {
                        cijk = 0;
                    }
                    double fac = cijk / (aij*ak*sqrt(aij+ak));
                    double theta_ij = ai * aj_aij;
                    double Kab = theta_ij * rjri[3*nsp_per_block];
                    gx[1536] = fac * exp(-Kab);
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, 4);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    double xjxi = rjri[0*nsp_per_block];  
                    double yjyi = rjri[1*nsp_per_block];  
                    double zjzi = rjri[2*nsp_per_block];  
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_ak = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gx[3072] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1536;
                        double xjxi = rjri[n * nsp_per_block];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[192] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 3 * b10 * s0;
                        _gx[256] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 4 * b10 * s0;
                        _gx[320] = s2;
                        double cpx = rt_ak * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[768] = s1;
                        s0 = _gx[64];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[832] = s1;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[64];
                        _gx[896] = s1;
                        s0 = _gx[192];
                        s1 = cpx * s0;
                        s1 += 3 * b00 * _gx[128];
                        _gx[960] = s1;
                        s0 = _gx[256];
                        s1 = cpx * s0;
                        s1 += 4 * b00 * _gx[192];
                        _gx[1024] = s1;
                        s0 = _gx[320];
                        s1 = cpx * s0;
                        s1 += 5 * b00 * _gx[256];
                        _gx[1088] = s1;
                        s1 = _gx[320];
                        s0 = _gx[256];
                        _gx[512] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[192];
                        _gx[448] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[384] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[320] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[256] = s1 - xjxi * s0;
                        s1 = _gx[512];
                        s0 = _gx[448];
                        _gx[704] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[384];
                        _gx[640] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[320];
                        _gx[576] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[256];
                        _gx[512] = s1 - xjxi * s0;
                        s1 = _gx[1088];
                        s0 = _gx[1024];
                        _gx[1280] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[960];
                        _gx[1216] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[896];
                        _gx[1152] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[832];
                        _gx[1088] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[768];
                        _gx[1024] = s1 - xjxi * s0;
                        s1 = _gx[1280];
                        s0 = _gx[1216];
                        _gx[1472] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1152];
                        _gx[1408] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1088];
                        _gx[1344] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1024];
                        _gx[1280] = s1 - xjxi * s0;
                    }
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        switch (gout_id) {
                        case 0:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[640];
                            Iy = gx[1536];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[1408] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += (aj2 * (gx[704] - xjxi * Ix) - 2 * gx[384]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1600];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[1280] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += (aj2 * (gx[576] - xjxi * Ix) - 2 * gx[256]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[320];
                            Iy = gx[1792];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[1088] * prod_yz;
                            v_ky += ak2 * gx[2560] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[12*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[384];
                            Iy = gx[1536];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[1152] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[4096] * prod_xy;
                            v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[16*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[256];
                            Iy = gx[1600];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += ak2 * gx[1024] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[4160] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[20*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[3*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[64];
                            Iy = gx[2048];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2816] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2112] - yjyi * Iy) - 2 * gx[1792]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[24*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[4*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[128];
                            Iy = gx[1792];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[896] * prod_yz;
                            v_ky += ak2 * gx[2560] * prod_xz;
                            v_kz += ak2 * gx[4096] * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[28*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[4*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[1856];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2624] * prod_xz;
                            v_kz += ak2 * gx[4160] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[32*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[5*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[64];
                            Iy = gx[1536];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3712] - 1 * gx[3584]) * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[4416] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3712] - zjzi * Iz) - 2 * gx[3392]) * prod_xy;
                            break;
                        case 1:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1600];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[1344] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 2 * gx[320]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1536];
                            Iz = gx[3200];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3264] - 2 * gx[3136]) * prod_xy;
                            v_kx += ak2 * gx[1280] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[3968] * prod_xy;
                            v_jx += (aj2 * (gx[576] - xjxi * Ix) - 2 * gx[256]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3264] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[9*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[256];
                            Iy = gx[1920];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1984] - 2 * gx[1856]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[1024] * prod_yz;
                            v_ky += ak2 * gx[2688] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += (aj2 * (gx[1984] - yjyi * Iy) - 1 * gx[1664]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[13*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[320];
                            Iy = gx[1600];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[1088] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[4096] * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[17*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[256];
                            Iy = gx[1536];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3520] - 2 * gx[3392]) * prod_xy;
                            v_kx += ak2 * gx[1024] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[4224] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3520] - zjzi * Iz) - 1 * gx[3200]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[21*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[3*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[2176];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2240] - 2 * gx[2112]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2944] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2240] - yjyi * Iy) - 2 * gx[1920]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[25*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[4*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[64];
                            Iy = gx[1856];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2624] * prod_xz;
                            v_kz += ak2 * gx[4096] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[29*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[4*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[1792];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3520] - 2 * gx[3392]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2560] * prod_xz;
                            v_kz += ak2 * gx[4224] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += (aj2 * (gx[3520] - zjzi * Iz) - 1 * gx[3200]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[33*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[5*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[1664];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2432] * prod_xz;
                            v_kz += ak2 * gx[4352] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3648] - zjzi * Iz) - 2 * gx[3328]) * prod_xy;
                            break;
                        case 2:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1536];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[1344] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 2 * gx[320]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[384];
                            Iy = gx[1792];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[1152] * prod_yz;
                            v_ky += ak2 * gx[2560] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[10*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[256];
                            Iy = gx[1856];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[1024] * prod_yz;
                            v_ky += ak2 * gx[2624] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[14*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[320];
                            Iy = gx[1536];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += ak2 * gx[1088] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[4160] * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[18*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[3*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[128];
                            Iy = gx[2048];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[896] * prod_yz;
                            v_ky += ak2 * gx[2816] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2112] - yjyi * Iy) - 2 * gx[1792]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[22*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[3*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[2112];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2880] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2176] - yjyi * Iy) - 2 * gx[1856]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[26*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[4*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[64];
                            Iy = gx[1792];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2560] * prod_xz;
                            v_kz += ak2 * gx[4160] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[30*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[5*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[128];
                            Iy = gx[1536];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[896] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[4352] * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3648] - zjzi * Iz) - 2 * gx[3328]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[34*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[5*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[1600];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[3712] - 1 * gx[3584]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[4416] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3712] - zjzi * Iz) - 2 * gx[3392]) * prod_xy;
                            break;
                        case 3:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1664];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[1280] * prod_yz;
                            v_ky += ak2 * gx[2432] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += (aj2 * (gx[576] - xjxi * Ix) - 2 * gx[256]) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[320];
                            Iy = gx[1856];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[1088] * prod_yz;
                            v_ky += ak2 * gx[2624] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[11*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[256];
                            Iy = gx[1792];
                            Iz = gx[3200];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3264] - 2 * gx[3136]) * prod_xy;
                            v_kx += ak2 * gx[1024] * prod_yz;
                            v_ky += ak2 * gx[2560] * prod_xz;
                            v_kz += ak2 * gx[3968] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3264] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[15*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[256];
                            Iy = gx[1664];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[1024] * prod_yz;
                            v_ky += ak2 * gx[2432] * prod_xz;
                            v_kz += ak2 * gx[4096] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[19*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[3*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[64];
                            Iy = gx[2112];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2880] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2176] - yjyi * Iy) - 2 * gx[1856]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[23*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[3*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[2048];
                            Iz = gx[3200];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += (ai2 * gx[3264] - 2 * gx[3136]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2816] * prod_xz;
                            v_kz += ak2 * gx[3968] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2112] - yjyi * Iy) - 2 * gx[1792]) * prod_xz;
                            v_jz += aj2 * (gx[3264] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[27*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[4*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[1920];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1984] - 2 * gx[1856]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2688] * prod_xz;
                            v_kz += ak2 * gx[4096] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1984] - yjyi * Iy) - 1 * gx[1664]) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[31*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[5*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[64];
                            Iy = gx[1600];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[4352] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3648] - zjzi * Iz) - 2 * gx[3328]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[35*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[5*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[0];
                            Iy = gx[1536];
                            Iz = gx[3712];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3776] - 2 * gx[3648]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[4480] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3776] - zjzi * Iz) - 2 * gx[3456]) * prod_xy;
                            break;
                        }
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
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                        }
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_11x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = trr_01x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                        }
                        Ix = trr_10x;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+1];
                        }
                        Ix = 1;
                        Iy = trr_11y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+1];
                        }
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                        }
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+2];
                        }
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+2];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_11z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
void int3c2e_ip1_111(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    constexpr int aux_per_block = 16;
    constexpr int nsp_per_block = 4;
    constexpr int nst_per_block = 64;
    int gout_id = thread_id / 64;
    int st_id = thread_id % 64;
    int sp_id = st_id / aux_per_block;
    int aux_id = st_id - sp_id * aux_per_block;
    extern __shared__ double shared_memory[];
    double *Rpq = shared_memory + st_id;
    double *gx = shared_memory + 192 + st_id;
    double *rw = shared_memory + 3648 + st_id;
    double *rjri = shared_memory + 3648 + nroots * 128 + sp_id;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
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
        if (gout_id == 0 && aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
            rjri[3*nsp_per_block] = rr_ij;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double s0, s1, s2;
            double dm_ijk;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
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
                __syncthreads();
                double xij = rjri[0*nsp_per_block] * aj_aij + ri[0];
                double yij = rjri[1*nsp_per_block] * aj_aij + ri[1];
                double zij = rjri[2*nsp_per_block] * aj_aij + ri[2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                if (gout_id == 0) {
                    double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                    if (ish == jsh) {
                        cijk *= .5;
                    } else if (ish < jsh) {
                        cijk = 0;
                    }
                    double fac = cijk / (aij*ak*sqrt(aij+ak));
                    double theta_ij = ai * aj_aij;
                    double Kab = theta_ij * rjri[3*nsp_per_block];
                    gx[1152] = fac * exp(-Kab);
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, 4);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    double xjxi = rjri[0*nsp_per_block];  
                    double yjyi = rjri[1*nsp_per_block];  
                    double zjzi = rjri[2*nsp_per_block];  
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_ak = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/ak * (1 - rt_ak);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gx[2304] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1152;
                        double xjxi = rjri[n * nsp_per_block];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[192] = s2;
                        double cpx = rt_ak * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[384] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        _gx[768] = s2;
                        s0 = _gx[64];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[448] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 1 * b00 * _gx[384];
                        _gx[832] = s2;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[64];
                        _gx[512] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 2 * b00 * _gx[448];
                        _gx[896] = s2;
                        s0 = _gx[192];
                        s1 = cpx * s0;
                        s1 += 3 * b00 * _gx[128];
                        _gx[576] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 3 * b00 * _gx[512];
                        _gx[960] = s2;
                        s1 = _gx[192];
                        s0 = _gx[128];
                        _gx[320] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[256] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[192] = s1 - xjxi * s0;
                        s1 = _gx[576];
                        s0 = _gx[512];
                        _gx[704] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[448];
                        _gx[640] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[384];
                        _gx[576] = s1 - xjxi * s0;
                        s1 = _gx[960];
                        s0 = _gx[896];
                        _gx[1088] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[832];
                        _gx[1024] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[768];
                        _gx[960] = s1 - xjxi * s0;
                    }
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        switch (gout_id) {
                        case 0:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[640];
                            Iy = gx[1152];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[704] - 1 * gx[576]) * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += (aj2 * (gx[704] - xjxi * Ix) - 1 * gx[448]) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[384];
                            Iy = gx[1408];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += (ai2 * gx[1472] - 1 * gx[1344]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1792] * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1472] - yjyi * Iy) - 1 * gx[1216]) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[384];
                            Iy = gx[1152];
                            Iz = gx[2560];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += (ai2 * gx[2624] - 1 * gx[2496]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += ak2 * gx[2944] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2624] - zjzi * Iz) - 1 * gx[2368]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[64];
                            Iy = gx[1728];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += (ak2 * gx[2112] - 1 * gx[1344]) * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[1600];
                            Iz = gx[2496];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[2560] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[1984] - 1 * gx[1216]) * prod_xz;
                            v_kz += ak2 * gx[2880] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2560] - zjzi * Iz) - 1 * gx[2304]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[192];
                            Iy = gx[1152];
                            Iz = gx[2752];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += (ai2 * gx[2816] - 1 * gx[2688]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += (ak2 * gx[3136] - 1 * gx[2368]) * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2816] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[64];
                            Iy = gx[1152];
                            Iz = gx[2880];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += ai2 * gx[2944] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += (ak2 * gx[3264] - 1 * gx[2496]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2944] - zjzi * Iz) - 1 * gx[2688]) * prod_xy;
                            break;
                        case 1:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1216];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[640] * prod_yz;
                            v_iy += (ai2 * gx[1280] - 1 * gx[1152]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += (ak2 * gx[960] - 1 * gx[192]) * prod_yz;
                            v_ky += ak2 * gx[1600] * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 1 * gx[384]) * prod_yz;
                            v_jy += aj2 * (gx[1280] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[384];
                            Iy = gx[1344];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += ai2 * gx[1408] * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1728] * prod_xz;
                            v_kz += ak2 * gx[2752] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1408] - yjyi * Iy) - 1 * gx[1152]) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[256];
                            Iy = gx[1536];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[320] - 1 * gx[192]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += (ak2 * gx[1920] - 1 * gx[1152]) * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[1792];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1856] - 1 * gx[1728]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2176] - 1 * gx[1408]) * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[1536];
                            Iz = gx[2560];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[2624] - 1 * gx[2496]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[1920] - 1 * gx[1152]) * prod_xz;
                            v_kz += ak2 * gx[2944] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2624] - zjzi * Iz) - 1 * gx[2368]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[64];
                            Iy = gx[1344];
                            Iz = gx[2688];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1408] * prod_xz;
                            v_iz += ai2 * gx[2752] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += ak2 * gx[1728] * prod_xz;
                            v_kz += (ak2 * gx[3072] - 1 * gx[2304]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1408] - yjyi * Iy) - 1 * gx[1152]) * prod_xz;
                            v_jz += aj2 * (gx[2752] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1216];
                            Iz = gx[2880];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1280] - 1 * gx[1152]) * prod_xz;
                            v_iz += ai2 * gx[2944] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[1600] * prod_xz;
                            v_kz += (ak2 * gx[3264] - 1 * gx[2496]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1280] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2944] - zjzi * Iz) - 1 * gx[2688]) * prod_xy;
                            break;
                        case 2:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1152];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[640] * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += (ak2 * gx[960] - 1 * gx[192]) * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += ak2 * gx[2752] * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 1 * gx[384]) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[448];
                            Iy = gx[1152];
                            Iz = gx[2496];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[512] - 1 * gx[384]) * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += ai2 * gx[2560] * prod_xy;
                            v_kx += (ak2 * gx[832] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += ak2 * gx[2880] * prod_xy;
                            v_jx += aj2 * (gx[512] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2560] - zjzi * Iz) - 1 * gx[2304]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[192];
                            Iy = gx[1600];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[1984] - 1 * gx[1216]) * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[1728];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2112] - 1 * gx[1344]) * prod_xz;
                            v_kz += ak2 * gx[2752] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[256];
                            Iy = gx[1152];
                            Iz = gx[2688];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[320] - 1 * gx[192]) * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += ai2 * gx[2752] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += (ak2 * gx[3072] - 1 * gx[2304]) * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2752] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1408];
                            Iz = gx[2688];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1472] - 1 * gx[1344]) * prod_xz;
                            v_iz += ai2 * gx[2752] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[1792] * prod_xz;
                            v_kz += (ak2 * gx[3072] - 1 * gx[2304]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1472] - yjyi * Iy) - 1 * gx[1216]) * prod_xz;
                            v_jz += aj2 * (gx[2752] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1152];
                            Iz = gx[2944];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1216] * prod_xz;
                            v_iz += (ai2 * gx[3008] - 1 * gx[2880]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[1536] * prod_xz;
                            v_kz += (ak2 * gx[3328] - 1 * gx[2560]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3008] - zjzi * Iz) - 1 * gx[2752]) * prod_xy;
                            break;
                        case 3:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[448];
                            Iy = gx[1344];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[512] - 1 * gx[384]) * prod_yz;
                            v_iy += ai2 * gx[1408] * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += (ak2 * gx[832] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[1728] * prod_xz;
                            v_kz += ak2 * gx[2688] * prod_xy;
                            v_jx += aj2 * (gx[512] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1408] - yjyi * Iy) - 1 * gx[1152]) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[384];
                            Iy = gx[1216];
                            Iz = gx[2496];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += (ai2 * gx[1280] - 1 * gx[1152]) * prod_xz;
                            v_iz += ai2 * gx[2560] * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1600] * prod_xz;
                            v_kz += ak2 * gx[2880] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1280] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2560] - zjzi * Iz) - 1 * gx[2304]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[192];
                            Iy = gx[1536];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[1920] - 1 * gx[1152]) * prod_xz;
                            v_kz += ak2 * gx[2752] * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[64];
                            Iy = gx[1536];
                            Iz = gx[2496];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[2560] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += (ak2 * gx[1920] - 1 * gx[1152]) * prod_xz;
                            v_kz += ak2 * gx[2880] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[2560] - zjzi * Iz) - 1 * gx[2304]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[192];
                            Iy = gx[1216];
                            Iz = gx[2688];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += (ai2 * gx[1280] - 1 * gx[1152]) * prod_xz;
                            v_iz += ai2 * gx[2752] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[1600] * prod_xz;
                            v_kz += (ak2 * gx[3072] - 1 * gx[2304]) * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1280] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2752] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1344];
                            Iz = gx[2752];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1408] * prod_xz;
                            v_iz += (ai2 * gx[2816] - 1 * gx[2688]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[1728] * prod_xz;
                            v_kz += (ak2 * gx[3136] - 1 * gx[2368]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1408] - yjyi * Iy) - 1 * gx[1152]) * prod_xz;
                            v_jz += aj2 * (gx[2816] - zjzi * Iz) * prod_xy;
                            break;
                        }
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
void int3c2e_ip1_201(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_21x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        fxi = ai2 * trr_31x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        fxi -= 2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_211x = trr_31x - xjxi * trr_21x;
                        fxj = aj2 * hrr_211x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        fxk = ak2 * trr_22x;
                        fxk -= 1 * trr_20x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = trr_11x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_21x;
                        double trr_01x = cpx * 1;
                        fxi -= 1 * trr_01x;
                        v_ix += fxi * prod_yz;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_111x = trr_21x - xjxi * trr_11x;
                        fxj = aj2 * hrr_111x;
                        v_jx += fxj * prod_yz;
                        double hrr_110y = trr_20y - yjyi * trr_10y;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        fxk = ak2 * trr_12x;
                        fxk -= 1 * trr_10x;
                        v_kx += fxk * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = trr_11x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_21x;
                        fxi -= 1 * trr_01x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_111x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_110z = trr_20z - zjzi * trr_10z;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_12x;
                        fxk -= 1 * trr_10x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[3*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+0];
                        }
                        Ix = trr_01x;
                        Iy = trr_20y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        fyi = ai2 * trr_30y;
                        fyi -= 2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_011x = trr_11x - xjxi * trr_01x;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        double hrr_210y = trr_30y - yjyi * trr_20y;
                        fyj = aj2 * hrr_210y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        fyk = ak2 * trr_21y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[4*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+0];
                        }
                        Ix = trr_01x;
                        Iy = trr_10y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[5*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+0];
                        }
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = trr_20z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        fzi = ai2 * trr_30z;
                        fzi -= 2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_210z = trr_30z - zjzi * trr_20z;
                        fzj = aj2 * hrr_210z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        fzk = ak2 * trr_21z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                        }
                        Ix = trr_20x;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_30x;
                        fxi -= 2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_210x = trr_30x - xjxi * trr_20x;
                        fxj = aj2 * hrr_210x;
                        v_jx += fxj * prod_yz;
                        double hrr_011y = trr_11y - yjyi * trr_01y;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_21x;
                        v_kx += fxk * prod_yz;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+1];
                        }
                        Ix = trr_10x;
                        Iy = trr_11y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_21y;
                        fyi -= 1 * trr_01y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_110x = trr_20x - xjxi * trr_10x;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        double hrr_111y = trr_21y - yjyi * trr_11y;
                        fyj = aj2 * hrr_111y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        fyk = ak2 * trr_12y;
                        fyk -= 1 * trr_10y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+1];
                        }
                        Ix = trr_10x;
                        Iy = trr_01y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[3*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+1];
                        }
                        Ix = 1;
                        Iy = trr_21y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        fyi = ai2 * trr_31y;
                        fyi -= 2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_211y = trr_31y - yjyi * trr_21y;
                        fyj = aj2 * hrr_211y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        fyk = ak2 * trr_22y;
                        fyk -= 1 * trr_20y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[4*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+1];
                        }
                        Ix = 1;
                        Iy = trr_11y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_21y;
                        fyi -= 1 * trr_01y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_111y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_12y;
                        fyk -= 1 * trr_10y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[5*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+1];
                        }
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = trr_20z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_30z;
                        fzi -= 2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_210z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_21z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                        }
                        Ix = trr_20x;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_30x;
                        fxi -= 2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_210x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_011z = trr_11z - zjzi * trr_01z;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_21x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+2];
                        }
                        Ix = trr_10x;
                        Iy = trr_10y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+2];
                        }
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = trr_11z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_21z;
                        fzi -= 1 * trr_01z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_111z = trr_21z - zjzi * trr_11z;
                        fzj = aj2 * hrr_111z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        fzk = ak2 * trr_12z;
                        fzk -= 1 * trr_10z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[3*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+2];
                        }
                        Ix = 1;
                        Iy = trr_20y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_30y;
                        fyi -= 2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_210y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_21y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[4*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+2];
                        }
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = trr_11z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_21z;
                        fzi -= 1 * trr_01z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_111z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_12z;
                        fzk -= 1 * trr_10z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[5*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+2];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_21z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        fzi = ai2 * trr_31z;
                        fzi -= 2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_211z = trr_31z - zjzi * trr_21z;
                        fzj = aj2 * hrr_211z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        fzk = ak2 * trr_22z;
                        fzk -= 1 * trr_20z;
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
void int3c2e_ip1_211(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    constexpr int aux_per_block = 16;
    constexpr int nsp_per_block = 4;
    constexpr int nst_per_block = 64;
    int gout_id = thread_id / 64;
    int st_id = thread_id % 64;
    int sp_id = st_id / aux_per_block;
    int aux_id = st_id - sp_id * aux_per_block;
    extern __shared__ double shared_memory[];
    double *Rpq = shared_memory + st_id;
    double *gx = shared_memory + 192 + st_id;
    double *rw = shared_memory + 4800 + st_id;
    double *rjri = shared_memory + 4800 + nroots * 128 + sp_id;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
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
        if (gout_id == 0 && aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
            rjri[3*nsp_per_block] = rr_ij;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double s0, s1, s2;
            double dm_ijk;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
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
                __syncthreads();
                double xij = rjri[0*nsp_per_block] * aj_aij + ri[0];
                double yij = rjri[1*nsp_per_block] * aj_aij + ri[1];
                double zij = rjri[2*nsp_per_block] * aj_aij + ri[2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                if (gout_id == 0) {
                    double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                    if (ish == jsh) {
                        cijk *= .5;
                    } else if (ish < jsh) {
                        cijk = 0;
                    }
                    double fac = cijk / (aij*ak*sqrt(aij+ak));
                    double theta_ij = ai * aj_aij;
                    double Kab = theta_ij * rjri[3*nsp_per_block];
                    gx[1536] = fac * exp(-Kab);
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, 4);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    double xjxi = rjri[0*nsp_per_block];  
                    double yjyi = rjri[1*nsp_per_block];  
                    double zjzi = rjri[2*nsp_per_block];  
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_ak = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/ak * (1 - rt_ak);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gx[3072] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1536;
                        double xjxi = rjri[n * nsp_per_block];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[192] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 3 * b10 * s0;
                        _gx[256] = s2;
                        double cpx = rt_ak * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[512] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        _gx[1024] = s2;
                        s0 = _gx[64];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[576] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 1 * b00 * _gx[512];
                        _gx[1088] = s2;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[64];
                        _gx[640] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 2 * b00 * _gx[576];
                        _gx[1152] = s2;
                        s0 = _gx[192];
                        s1 = cpx * s0;
                        s1 += 3 * b00 * _gx[128];
                        _gx[704] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 3 * b00 * _gx[640];
                        _gx[1216] = s2;
                        s0 = _gx[256];
                        s1 = cpx * s0;
                        s1 += 4 * b00 * _gx[192];
                        _gx[768] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 4 * b00 * _gx[704];
                        _gx[1280] = s2;
                        s1 = _gx[256];
                        s0 = _gx[192];
                        _gx[448] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[384] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[320] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[256] = s1 - xjxi * s0;
                        s1 = _gx[768];
                        s0 = _gx[704];
                        _gx[960] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[640];
                        _gx[896] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[576];
                        _gx[832] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[512];
                        _gx[768] = s1 - xjxi * s0;
                        s1 = _gx[1280];
                        s0 = _gx[1216];
                        _gx[1472] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1152];
                        _gx[1408] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1088];
                        _gx[1344] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1024];
                        _gx[1280] = s1 - xjxi * s0;
                    }
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        switch (gout_id) {
                        case 0:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[896];
                            Iy = gx[1536];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[960] - 2 * gx[832]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1408] - 1 * gx[384]) * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += (aj2 * (gx[960] - xjxi * Ix) - 1 * gx[640]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[768];
                            Iy = gx[1600];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[832] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[1280] - 1 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += (aj2 * (gx[832] - xjxi * Ix) - 1 * gx[512]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1792];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[12*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[640];
                            Iy = gx[1536];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += (ak2 * gx[1152] - 1 * gx[128]) * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[704] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[16*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1600];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[320];
                            Iy = gx[2048];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[2112] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[128];
                            Iy = gx[2304];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[2368] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += (ak2 * gx[2816] - 1 * gx[1792]) * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2368] - yjyi * Iy) - 1 * gx[2048]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[10*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+4] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[2368];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += (ak2 * gx[2880] - 1 * gx[1856]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2432] - yjyi * Iy) - 1 * gx[2112]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[14*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[64];
                            Iy = gx[2048];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2112] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[384];
                            Iy = gx[1536];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[896] * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3072]) * prod_xy;
                            v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3648] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+2];
                            }
                            Ix = gx[256];
                            Iy = gx[1600];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[3712] - 1 * gx[3584]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[4160] - 1 * gx[3136]) * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3712] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[64];
                            Iy = gx[1792];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3712] - 1 * gx[3584]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += (ak2 * gx[4160] - 1 * gx[3136]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3712] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[12*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[128];
                            Iy = gx[1536];
                            Iz = gx[3840];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3904] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += (ak2 * gx[4352] - 1 * gx[3328]) * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3904] - zjzi * Iz) - 1 * gx[3584]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[16*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+4] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1600];
                            Iz = gx[3904];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[3968] - 1 * gx[3840]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[4416] - 1 * gx[3392]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3968] - zjzi * Iz) - 1 * gx[3648]) * prod_xy;
                            break;
                        case 1:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[832];
                            Iy = gx[1600];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1344] - 1 * gx[320]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += (aj2 * (gx[896] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[768];
                            Iy = gx[1536];
                            Iz = gx[3200];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[832] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3264] - 2 * gx[3136]) * prod_xy;
                            v_kx += (ak2 * gx[1280] - 1 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += ak2 * gx[3712] * prod_xy;
                            v_jx += (aj2 * (gx[832] - xjxi * Ix) - 1 * gx[512]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3264] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[9*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1920];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1984] - 2 * gx[1856]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2432] * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1984] - yjyi * Iy) - 1 * gx[1664]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[13*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1600];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[17*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1536];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3520] - 2 * gx[3392]) * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += ak2 * gx[3968] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3520] - zjzi * Iz) - 1 * gx[3200]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+1];
                            }
                            Ix = gx[256];
                            Iy = gx[2176];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[2240] - 2 * gx[2112]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += (ak2 * gx[2688] - 1 * gx[1664]) * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[2240] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[64];
                            Iy = gx[2368];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2880] - 1 * gx[1856]) * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2432] - yjyi * Iy) - 1 * gx[2112]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[11*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+5] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[2304];
                            Iz = gx[3200];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[2368] * prod_xz;
                            v_iz += (ai2 * gx[3264] - 2 * gx[3136]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += (ak2 * gx[2816] - 1 * gx[1792]) * prod_xz;
                            v_kz += ak2 * gx[3712] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2368] - yjyi * Iy) - 1 * gx[2048]) * prod_xz;
                            v_jz += aj2 * (gx[3264] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[15*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+3] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[2176];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2240] - 2 * gx[2112]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += (ak2 * gx[2688] - 1 * gx[1664]) * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2240] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[320];
                            Iy = gx[1600];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3072]) * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3648] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+2];
                            }
                            Ix = gx[256];
                            Iy = gx[1536];
                            Iz = gx[3712];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3776] - 2 * gx[3648]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += (ak2 * gx[4224] - 1 * gx[3200]) * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3776] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[9*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+3] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1920];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1984] - 2 * gx[1856]) * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += ak2 * gx[2432] * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3072]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1984] - yjyi * Iy) - 1 * gx[1664]) * prod_xz;
                            v_jz += aj2 * (gx[3648] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[13*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[64];
                            Iy = gx[1600];
                            Iz = gx[3840];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3904] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[4352] - 1 * gx[3328]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3904] - zjzi * Iz) - 1 * gx[3584]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[17*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+5] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1536];
                            Iz = gx[3968];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[4032] - 2 * gx[3904]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += (ak2 * gx[4480] - 1 * gx[3456]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[4032] - zjzi * Iz) - 1 * gx[3712]) * prod_xy;
                            break;
                        case 2:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[832];
                            Iy = gx[1536];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[1344] - 1 * gx[320]) * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += (aj2 * (gx[896] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[640];
                            Iy = gx[1792];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1152] - 1 * gx[128]) * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += aj2 * (gx[704] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[10*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1856];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[14*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1536];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[384];
                            Iy = gx[2048];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[896] * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                            v_jy += aj2 * (gx[2112] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+1];
                            }
                            Ix = gx[256];
                            Iy = gx[2112];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += (ak2 * gx[2624] - 1 * gx[1600]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[2176] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[64];
                            Iy = gx[2304];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[2368] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2816] - 1 * gx[1792]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2368] - yjyi * Iy) - 1 * gx[2048]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[12*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[128];
                            Iy = gx[2048];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2112] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[16*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+4] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[2112];
                            Iz = gx[3392];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xz;
                            v_iz += (ai2 * gx[3456] - 1 * gx[3328]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += (ak2 * gx[2624] - 1 * gx[1600]) * prod_xz;
                            v_kz += ak2 * gx[3904] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2176] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3456] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[320];
                            Iy = gx[1536];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3712] - 1 * gx[3584]) * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += (ak2 * gx[4160] - 1 * gx[3136]) * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3712] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[128];
                            Iy = gx[1792];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3072]) * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3648] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[10*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+4] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1856];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += (ai2 * gx[3712] - 1 * gx[3584]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += (ak2 * gx[4160] - 1 * gx[3136]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3712] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[14*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[64];
                            Iy = gx[1536];
                            Iz = gx[3904];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3968] - 1 * gx[3840]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[2048] * prod_xz;
                            v_kz += (ak2 * gx[4416] - 1 * gx[3392]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3968] - zjzi * Iz) - 1 * gx[3648]) * prod_xy;
                            break;
                        case 3:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[768];
                            Iy = gx[1664];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[832] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1280] - 1 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[2176] * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += (aj2 * (gx[832] - xjxi * Ix) - 1 * gx[512]) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1856];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[11*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1792];
                            Iz = gx[3200];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3264] - 2 * gx[3136]) * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += ak2 * gx[3712] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3264] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[15*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1664];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2176] * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[320];
                            Iy = gx[2112];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[832] * prod_yz;
                            v_ky += (ak2 * gx[2624] - 1 * gx[1600]) * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[2176] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+1];
                            }
                            Ix = gx[256];
                            Iy = gx[2048];
                            Iz = gx[3200];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += (ai2 * gx[3264] - 2 * gx[3136]) * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3712] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[2112] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3264] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[9*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+3] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[2432];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2496] - 2 * gx[2368]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += (ak2 * gx[2944] - 1 * gx[1920]) * prod_xz;
                            v_kz += ak2 * gx[3584] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2496] - yjyi * Iy) - 1 * gx[2176]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[13*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[64];
                            Iy = gx[2112];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xz;
                            v_iz += ai2 * gx[3392] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2624] - 1 * gx[1600]) * prod_xz;
                            v_kz += ak2 * gx[3840] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2176] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[17*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+5] * density_auxvec[k0+1];
                            }
                            Ix = gx[0];
                            Iy = gx[2048];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[2112] * prod_xz;
                            v_iz += (ai2 * gx[3520] - 2 * gx[3392]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3968] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2112] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3520] - zjzi * Iz) - 1 * gx[3200]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+2];
                            }
                            Ix = gx[256];
                            Iy = gx[1664];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[768] * prod_yz;
                            v_ky += ak2 * gx[2176] * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3072]) * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3648] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[64];
                            Iy = gx[1856];
                            Iz = gx[3584];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[1920] - 1 * gx[1792]) * prod_xz;
                            v_iz += ai2 * gx[3648] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[2368] * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3072]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1920] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3648] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[11*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+5] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1792];
                            Iz = gx[3712];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1856] * prod_xz;
                            v_iz += (ai2 * gx[3776] - 2 * gx[3648]) * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += ak2 * gx[2304] * prod_xz;
                            v_kz += (ak2 * gx[4224] - 1 * gx[3200]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3776] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[15*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+3] * density_auxvec[k0+2];
                            }
                            Ix = gx[0];
                            Iy = gx[1664];
                            Iz = gx[3840];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[3904] * prod_xy;
                            v_kx += ak2 * gx[512] * prod_yz;
                            v_ky += ak2 * gx[2176] * prod_xz;
                            v_kz += (ak2 * gx[4352] - 1 * gx[3328]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3904] - zjzi * Iz) - 1 * gx[3584]) * prod_xy;
                            break;
                        }
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
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_02x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                        }
                        Ix = trr_01x;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                        }
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 3*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+3];
                        }
                        Ix = 1;
                        Iy = trr_02y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 4*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+4];
                        }
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 5*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+5];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_02z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
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
void int3c2e_ip1_102(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int sp_id = thread_id / BLOCK_SIZE;
    int aux_id = thread_id % BLOCK_SIZE;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double shared_memory[];
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
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double dm_ijk;
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
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                        }
                        Ix = trr_12x;
                        Iy = fac1;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        fxi = ai2 * trr_22x;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        fxi -= 1 * trr_02x;
                        v_ix += fxi * prod_yz;
                        double c0y = yjyi * aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac1;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double c0z = zjzi * aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_112x = trr_22x - xjxi * trr_12x;
                        fxj = aj2 * hrr_112x;
                        v_jx += fxj * prod_yz;
                        double hrr_010y = trr_10y - yjyi * fac1;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_010z = trr_10z - zjzi * wt;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        fxk = ak2 * trr_13x;
                        fxk -= 2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double cpy = ypq*rt_ak;
                        double trr_01y = cpy * fac1;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double cpz = zpq*rt_ak;
                        double trr_01z = cpz * wt;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                        }
                        Ix = trr_02x;
                        Iy = trr_10y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_12x;
                        v_ix += fxi * prod_yz;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_012x = trr_12x - xjxi * trr_02x;
                        fxj = aj2 * hrr_012x;
                        v_jx += fxj * prod_yz;
                        double hrr_110y = trr_20y - yjyi * trr_10y;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        fxk = ak2 * trr_03x;
                        fxk -= 2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 0*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                        }
                        Ix = trr_02x;
                        Iy = fac1;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_12x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_012x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_110z = trr_20z - zjzi * trr_10z;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_03x;
                        fxk -= 2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                        }
                        Ix = trr_11x;
                        Iy = trr_01y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_21x;
                        fxi -= 1 * trr_01x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_111x = trr_21x - xjxi * trr_11x;
                        fxj = aj2 * hrr_111x;
                        v_jx += fxj * prod_yz;
                        double hrr_011y = trr_11y - yjyi * trr_01y;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_12x;
                        fxk -= 1 * trr_10x;
                        v_kx += fxk * prod_yz;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+1];
                        }
                        Ix = trr_01x;
                        Iy = trr_11y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        fyi = ai2 * trr_21y;
                        fyi -= 1 * trr_01y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_011x = trr_11x - xjxi * trr_01x;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        double hrr_111y = trr_21y - yjyi * trr_11y;
                        fyj = aj2 * hrr_111y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        fyk = ak2 * trr_12y;
                        fyk -= 1 * trr_10y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 1*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+1];
                        }
                        Ix = trr_01x;
                        Iy = trr_01y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                        }
                        Ix = trr_11x;
                        Iy = fac1;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_21x;
                        fxi -= 1 * trr_01x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_111x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_011z = trr_11z - zjzi * trr_01z;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_12x;
                        fxk -= 1 * trr_10x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+2];
                        }
                        Ix = trr_01x;
                        Iy = trr_10y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 2*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+2];
                        }
                        Ix = trr_01x;
                        Iy = fac1;
                        Iz = trr_11z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_11x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        fzi = ai2 * trr_21z;
                        fzi -= 1 * trr_01z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_011x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_111z = trr_21z - zjzi * trr_11z;
                        fzj = aj2 * hrr_111z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_02x;
                        fxk -= 1 * 1;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        fzk = ak2 * trr_12z;
                        fzk -= 1 * trr_10z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 3*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+3];
                        }
                        Ix = trr_10x;
                        Iy = trr_02y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_12y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_110x = trr_20x - xjxi * trr_10x;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        double hrr_012y = trr_12y - yjyi * trr_02y;
                        fyj = aj2 * hrr_012y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        fyk = ak2 * trr_03y;
                        fyk -= 2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 3*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+3];
                        }
                        Ix = 1;
                        Iy = trr_12y;
                        Iz = wt;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        fyi = ai2 * trr_22y;
                        fyi -= 1 * trr_02y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_10z;
                        v_iz += fzi * prod_xy;
                        double hrr_010x = trr_10x - xjxi * 1;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        double hrr_112y = trr_22y - yjyi * trr_12y;
                        fyj = aj2 * hrr_112y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_010z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        fyk = ak2 * trr_13y;
                        fyk -= 2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 3*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+3];
                        }
                        Ix = 1;
                        Iy = trr_02y;
                        Iz = trr_10z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_12y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_20z;
                        fzi -= 1 * wt;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_012y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_110z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_03y;
                        fyk -= 2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_11z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 4*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+4];
                        }
                        Ix = trr_10x;
                        Iy = trr_01y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 4*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+4];
                        }
                        Ix = 1;
                        Iy = trr_11y;
                        Iz = trr_01z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_21y;
                        fyi -= 1 * trr_01y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_11z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_111y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_011z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_12y;
                        fyk -= 1 * trr_10y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_02z;
                        fzk -= 1 * wt;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 4*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+4];
                        }
                        Ix = 1;
                        Iy = trr_01y;
                        Iz = trr_11z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_11y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_21z;
                        fzi -= 1 * trr_01z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_011y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_111z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_02y;
                        fyk -= 1 * fac1;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_12z;
                        fzk -= 1 * trr_10z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[0*naux + 5*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+5];
                        }
                        Ix = trr_10x;
                        Iy = fac1;
                        Iz = trr_02z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_20x;
                        fxi -= 1 * 1;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_12z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_110x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_012z = trr_12z - zjzi * trr_02z;
                        fzj = aj2 * hrr_012z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_11x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        fzk = ak2 * trr_03z;
                        fzk -= 2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[1*naux + 5*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+5];
                        }
                        Ix = 1;
                        Iy = trr_10y;
                        Iz = trr_02z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_20y;
                        fyi -= 1 * fac1;
                        v_iy += fyi * prod_xz;
                        fzi = ai2 * trr_12z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_110y;
                        v_jy += fyj * prod_xz;
                        fzj = aj2 * hrr_012z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_11y;
                        v_ky += fyk * prod_xz;
                        fzk = ak2 * trr_03z;
                        fzk -= 2 * trr_01z;
                        v_kz += fzk * prod_xy;
                        if (density_auxvec == NULL) {
                            dm_ijk = dm_tensor[2*naux + 5*nksh];
                        } else {
                            dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+5];
                        }
                        Ix = 1;
                        Iy = fac1;
                        Iz = trr_12z;
                        prod_xy = Ix * Iy * dm_ijk;
                        prod_xz = Ix * Iz * dm_ijk;
                        prod_yz = Iy * Iz * dm_ijk;
                        fxi = ai2 * trr_10x;
                        v_ix += fxi * prod_yz;
                        fyi = ai2 * trr_10y;
                        v_iy += fyi * prod_xz;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        fzi = ai2 * trr_22z;
                        fzi -= 1 * trr_02z;
                        v_iz += fzi * prod_xy;
                        fxj = aj2 * hrr_010x;
                        v_jx += fxj * prod_yz;
                        fyj = aj2 * hrr_010y;
                        v_jy += fyj * prod_xz;
                        double hrr_112z = trr_22z - zjzi * trr_12z;
                        fzj = aj2 * hrr_112z;
                        v_jz += fzj * prod_xy;
                        fxk = ak2 * trr_01x;
                        v_kx += fxk * prod_yz;
                        fyk = ak2 * trr_01y;
                        v_ky += fyk * prod_xz;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        fzk = ak2 * trr_13z;
                        fzk -= 2 * trr_11z;
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
void int3c2e_ip1_112(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    constexpr int aux_per_block = 16;
    constexpr int nsp_per_block = 4;
    constexpr int nst_per_block = 64;
    int gout_id = thread_id / 64;
    int st_id = thread_id % 64;
    int sp_id = st_id / aux_per_block;
    int aux_id = st_id - sp_id * aux_per_block;
    extern __shared__ double shared_memory[];
    double *Rpq = shared_memory + st_id;
    double *gx = shared_memory + 192 + st_id;
    double *rw = shared_memory + 4800 + st_id;
    double *rjri = shared_memory + 4800 + nroots * 128 + sp_id;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
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
        if (gout_id == 0 && aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
            rjri[3*nsp_per_block] = rr_ij;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double s0, s1, s2;
            double dm_ijk;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
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
                __syncthreads();
                double xij = rjri[0*nsp_per_block] * aj_aij + ri[0];
                double yij = rjri[1*nsp_per_block] * aj_aij + ri[1];
                double zij = rjri[2*nsp_per_block] * aj_aij + ri[2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                if (gout_id == 0) {
                    double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                    if (ish == jsh) {
                        cijk *= .5;
                    } else if (ish < jsh) {
                        cijk = 0;
                    }
                    double fac = cijk / (aij*ak*sqrt(aij+ak));
                    double theta_ij = ai * aj_aij;
                    double Kab = theta_ij * rjri[3*nsp_per_block];
                    gx[1536] = fac * exp(-Kab);
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, 4);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    double xjxi = rjri[0*nsp_per_block];  
                    double yjyi = rjri[1*nsp_per_block];  
                    double zjzi = rjri[2*nsp_per_block];  
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_ak = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/ak * (1 - rt_ak);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gx[3072] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1536;
                        double xjxi = rjri[n * nsp_per_block];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[192] = s2;
                        double cpx = rt_ak * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[384] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        _gx[768] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        _gx[1152] = s2;
                        s0 = _gx[64];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[448] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 1 * b00 * _gx[384];
                        _gx[832] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 1 * b00 * _gx[768];
                        _gx[1216] = s2;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[64];
                        _gx[512] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 2 * b00 * _gx[448];
                        _gx[896] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 2 * b00 * _gx[832];
                        _gx[1280] = s2;
                        s0 = _gx[192];
                        s1 = cpx * s0;
                        s1 += 3 * b00 * _gx[128];
                        _gx[576] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 3 * b00 * _gx[512];
                        _gx[960] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 3 * b00 * _gx[896];
                        _gx[1344] = s2;
                        s1 = _gx[192];
                        s0 = _gx[128];
                        _gx[320] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[256] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[192] = s1 - xjxi * s0;
                        s1 = _gx[576];
                        s0 = _gx[512];
                        _gx[704] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[448];
                        _gx[640] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[384];
                        _gx[576] = s1 - xjxi * s0;
                        s1 = _gx[960];
                        s0 = _gx[896];
                        _gx[1088] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[832];
                        _gx[1024] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[768];
                        _gx[960] = s1 - xjxi * s0;
                        s1 = _gx[1344];
                        s0 = _gx[1280];
                        _gx[1472] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1216];
                        _gx[1408] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[1152];
                        _gx[1344] = s1 - xjxi * s0;
                    }
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        switch (gout_id) {
                        case 0:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[1024];
                            Iy = gx[1536];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[1088] - 1 * gx[960]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1408] - 2 * gx[640]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += (aj2 * (gx[1088] - xjxi * Ix) - 1 * gx[832]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[768];
                            Iy = gx[1792];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[832] * prod_yz;
                            v_iy += (ai2 * gx[1856] - 1 * gx[1728]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1152] - 2 * gx[384]) * prod_yz;
                            v_ky += ak2 * gx[2176] * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[768];
                            Iy = gx[1536];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[832] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3392] - 1 * gx[3264]) * prod_xy;
                            v_kx += (ak2 * gx[1152] - 2 * gx[384]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += ak2 * gx[3712] * prod_xy;
                            v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[448];
                            Iy = gx[2112];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[512] - 1 * gx[384]) * prod_yz;
                            v_iy += ai2 * gx[2176] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[832] - 1 * gx[64]) * prod_yz;
                            v_ky += (ak2 * gx[2496] - 1 * gx[1728]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += aj2 * (gx[512] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2176] - yjyi * Iy) - 1 * gx[1920]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[384];
                            Iy = gx[1984];
                            Iz = gx[3264];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += (ai2 * gx[2048] - 1 * gx[1920]) * prod_xz;
                            v_iz += ai2 * gx[3328] * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += (ak2 * gx[2368] - 1 * gx[1600]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2048] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3328] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[576];
                            Iy = gx[1536];
                            Iz = gx[3520];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[640] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3584] - 1 * gx[3456]) * prod_xy;
                            v_kx += (ak2 * gx[960] - 1 * gx[192]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[3904] - 1 * gx[3136]) * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 1 * gx[384]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3584] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[448];
                            Iy = gx[1536];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[512] - 1 * gx[384]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3712] * prod_xy;
                            v_kx += (ak2 * gx[832] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[4032] - 1 * gx[3264]) * prod_xy;
                            v_jx += aj2 * (gx[512] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3712] - zjzi * Iz) - 1 * gx[3456]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+3];
                            }
                            Ix = gx[192];
                            Iy = gx[2368];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2752] - 2 * gx[1984]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[2432] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+3];
                            }
                            Ix = gx[0];
                            Iy = gx[2496];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[2560] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2880] - 2 * gx[2112]) * prod_xz;
                            v_kz += ak2 * gx[3520] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2560] - yjyi * Iy) - 1 * gx[2304]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+4];
                            }
                            Ix = gx[256];
                            Iy = gx[1920];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[320] - 1 * gx[192]) * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+4];
                            }
                            Ix = gx[0];
                            Iy = gx[2176];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2240] - 1 * gx[2112]) * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1792]) * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2240] - yjyi * Iy) - 1 * gx[1984]) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+4];
                            }
                            Ix = gx[0];
                            Iy = gx[1920];
                            Iz = gx[3712];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += (ai2 * gx[3776] - 1 * gx[3648]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3328]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3776] - zjzi * Iz) - 1 * gx[3520]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+5];
                            }
                            Ix = gx[64];
                            Iy = gx[1728];
                            Iz = gx[3840];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += ai2 * gx[3904] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[4224] - 2 * gx[3456]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3904] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+5];
                            }
                            Ix = gx[0];
                            Iy = gx[1600];
                            Iz = gx[4032];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[4096] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[1984] * prod_xz;
                            v_kz += (ak2 * gx[4416] - 2 * gx[3648]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[4096] - zjzi * Iz) - 1 * gx[3840]) * prod_xy;
                            break;
                        case 1:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[960];
                            Iy = gx[1600];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[1024] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1344] - 2 * gx[576]) * prod_yz;
                            v_ky += ak2 * gx[1984] * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += (aj2 * (gx[1024] - xjxi * Ix) - 1 * gx[768]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[768];
                            Iy = gx[1728];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[832] * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[1152] - 2 * gx[384]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += ak2 * gx[3520] * prod_xy;
                            v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[640];
                            Iy = gx[1920];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[704] - 1 * gx[576]) * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[256]) * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += (aj2 * (gx[704] - xjxi * Ix) - 1 * gx[448]) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[384];
                            Iy = gx[2176];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += (ai2 * gx[2240] - 1 * gx[2112]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += (ak2 * gx[2560] - 1 * gx[1792]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2240] - yjyi * Iy) - 1 * gx[1984]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[384];
                            Iy = gx[1920];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += (ai2 * gx[3392] - 1 * gx[3264]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3712] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[448];
                            Iy = gx[1728];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[512] - 1 * gx[384]) * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += (ak2 * gx[832] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += aj2 * (gx[512] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[384];
                            Iy = gx[1600];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3712] * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1984] * prod_xz;
                            v_kz += (ak2 * gx[4032] - 1 * gx[3264]) * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3712] - zjzi * Iz) - 1 * gx[3456]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+3];
                            }
                            Ix = gx[192];
                            Iy = gx[2304];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += ai2 * gx[2368] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2688] - 2 * gx[1920]) * prod_xz;
                            v_kz += ak2 * gx[3520] * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[2368] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+3];
                            }
                            Ix = gx[64];
                            Iy = gx[2304];
                            Iz = gx[3264];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[2368] * prod_xz;
                            v_iz += ai2 * gx[3328] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += (ak2 * gx[2688] - 2 * gx[1920]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2368] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3328] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+4];
                            }
                            Ix = gx[192];
                            Iy = gx[1984];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += (ai2 * gx[2048] - 1 * gx[1920]) * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2368] - 1 * gx[1600]) * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[2048] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+4];
                            }
                            Ix = gx[0];
                            Iy = gx[2112];
                            Iz = gx[3520];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[2176] * prod_xz;
                            v_iz += (ai2 * gx[3584] - 1 * gx[3456]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2496] - 1 * gx[1728]) * prod_xz;
                            v_kz += (ak2 * gx[3904] - 1 * gx[3136]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2176] - yjyi * Iy) - 1 * gx[1920]) * prod_xz;
                            v_jz += aj2 * (gx[3584] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+5];
                            }
                            Ix = gx[256];
                            Iy = gx[1536];
                            Iz = gx[3840];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[320] - 1 * gx[192]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3904] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[4224] - 2 * gx[3456]) * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3904] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+5];
                            }
                            Ix = gx[0];
                            Iy = gx[1792];
                            Iz = gx[3840];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1856] - 1 * gx[1728]) * prod_xz;
                            v_iz += ai2 * gx[3904] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[2176] * prod_xz;
                            v_kz += (ak2 * gx[4224] - 2 * gx[3456]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3904] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+5];
                            }
                            Ix = gx[0];
                            Iy = gx[1536];
                            Iz = gx[4096];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[4160] - 1 * gx[4032]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[4480] - 2 * gx[3712]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[4160] - zjzi * Iz) - 1 * gx[3904]) * prod_xy;
                            break;
                        case 2:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[960];
                            Iy = gx[1536];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[1024] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[1344] - 2 * gx[576]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += ak2 * gx[3520] * prod_xy;
                            v_jx += (aj2 * (gx[1024] - xjxi * Ix) - 1 * gx[768]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[832];
                            Iy = gx[1536];
                            Iz = gx[3264];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3328] * prod_xy;
                            v_kx += (ak2 * gx[1216] - 2 * gx[448]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[896] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3328] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[576];
                            Iy = gx[1984];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[640] * prod_yz;
                            v_iy += (ai2 * gx[2048] - 1 * gx[1920]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[960] - 1 * gx[192]) * prod_yz;
                            v_ky += (ak2 * gx[2368] - 1 * gx[1600]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 1 * gx[384]) * prod_yz;
                            v_jy += aj2 * (gx[2048] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[384];
                            Iy = gx[2112];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += ai2 * gx[2176] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += (ak2 * gx[2496] - 1 * gx[1728]) * prod_xz;
                            v_kz += ak2 * gx[3520] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2176] - yjyi * Iy) - 1 * gx[1920]) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[640];
                            Iy = gx[1536];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[704] - 1 * gx[576]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += (ak2 * gx[1024] - 1 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += (aj2 * (gx[704] - xjxi * Ix) - 1 * gx[448]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[384];
                            Iy = gx[1792];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += (ai2 * gx[1856] - 1 * gx[1728]) * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2176] * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1856] - yjyi * Iy) - 1 * gx[1600]) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[384];
                            Iy = gx[1536];
                            Iz = gx[3712];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3776] - 1 * gx[3648]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[4096] - 1 * gx[3328]) * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3776] - zjzi * Iz) - 1 * gx[3520]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+3];
                            }
                            Ix = gx[64];
                            Iy = gx[2496];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[2560] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += (ak2 * gx[2880] - 2 * gx[2112]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2560] - yjyi * Iy) - 1 * gx[2304]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+3];
                            }
                            Ix = gx[0];
                            Iy = gx[2368];
                            Iz = gx[3264];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xz;
                            v_iz += ai2 * gx[3328] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2752] - 2 * gx[1984]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2432] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3328] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+4];
                            }
                            Ix = gx[192];
                            Iy = gx[1920];
                            Iz = gx[3520];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += (ai2 * gx[3584] - 1 * gx[3456]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += (ak2 * gx[3904] - 1 * gx[3136]) * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3584] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+4];
                            }
                            Ix = gx[64];
                            Iy = gx[1920];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += ai2 * gx[3712] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += (ak2 * gx[4032] - 1 * gx[3264]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3712] - zjzi * Iz) - 1 * gx[3456]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+5];
                            }
                            Ix = gx[192];
                            Iy = gx[1600];
                            Iz = gx[3840];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3904] * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[1984] * prod_xz;
                            v_kz += (ak2 * gx[4224] - 2 * gx[3456]) * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3904] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+5];
                            }
                            Ix = gx[0];
                            Iy = gx[1728];
                            Iz = gx[3904];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += (ai2 * gx[3968] - 1 * gx[3840]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[4288] - 2 * gx[3520]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3968] - zjzi * Iz) * prod_xy;
                            break;
                        case 3:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[832];
                            Iy = gx[1728];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += (ak2 * gx[1216] - 2 * gx[448]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += aj2 * (gx[896] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[768];
                            Iy = gx[1600];
                            Iz = gx[3264];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[832] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3328] * prod_xy;
                            v_kx += (ak2 * gx[1152] - 2 * gx[384]) * prod_yz;
                            v_ky += ak2 * gx[1984] * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3328] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[576];
                            Iy = gx[1920];
                            Iz = gx[3136];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[640] * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += (ai2 * gx[3200] - 1 * gx[3072]) * prod_xy;
                            v_kx += (ak2 * gx[960] - 1 * gx[192]) * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3520] * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 1 * gx[384]) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3200] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[448];
                            Iy = gx[1920];
                            Iz = gx[3264];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[512] - 1 * gx[384]) * prod_yz;
                            v_iy += ai2 * gx[1984] * prod_xz;
                            v_iz += ai2 * gx[3328] * prod_xy;
                            v_kx += (ak2 * gx[832] - 1 * gx[64]) * prod_yz;
                            v_ky += (ak2 * gx[2304] - 1 * gx[1536]) * prod_xz;
                            v_kz += ak2 * gx[3648] * prod_xy;
                            v_jx += aj2 * (gx[512] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1984] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3328] - zjzi * Iz) - 1 * gx[3072]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[576];
                            Iy = gx[1600];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[640] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += (ak2 * gx[960] - 1 * gx[192]) * prod_yz;
                            v_ky += ak2 * gx[1984] * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += (aj2 * (gx[640] - xjxi * Ix) - 1 * gx[384]) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[384];
                            Iy = gx[1728];
                            Iz = gx[3520];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[448] * prod_yz;
                            v_iy += ai2 * gx[1792] * prod_xz;
                            v_iz += (ai2 * gx[3584] - 1 * gx[3456]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[2112] * prod_xz;
                            v_kz += (ak2 * gx[3904] - 1 * gx[3136]) * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[1792] - yjyi * Iy) - 1 * gx[1536]) * prod_xz;
                            v_jz += aj2 * (gx[3584] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+3];
                            }
                            Ix = gx[256];
                            Iy = gx[2304];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[320] - 1 * gx[192]) * prod_yz;
                            v_iy += ai2 * gx[2368] * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[640] * prod_yz;
                            v_ky += (ak2 * gx[2688] - 2 * gx[1920]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                            v_jy += aj2 * (gx[2368] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+1] * density_auxvec[k0+3];
                            }
                            Ix = gx[0];
                            Iy = gx[2560];
                            Iz = gx[3072];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2624] - 1 * gx[2496]) * prod_xz;
                            v_iz += ai2 * gx[3136] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2944] - 2 * gx[2176]) * prod_xz;
                            v_kz += ak2 * gx[3456] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2624] - yjyi * Iy) - 1 * gx[2368]) * prod_xz;
                            v_jz += aj2 * (gx[3136] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[8*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+2] * density_auxvec[k0+3];
                            }
                            Ix = gx[0];
                            Iy = gx[2304];
                            Iz = gx[3328];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[2368] * prod_xz;
                            v_iz += (ai2 * gx[3392] - 1 * gx[3264]) * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2688] - 2 * gx[1920]) * prod_xz;
                            v_kz += ak2 * gx[3712] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2368] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3392] - zjzi * Iz) - 1 * gx[3136]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[1*nao+0] * density_auxvec[k0+4];
                            }
                            Ix = gx[64];
                            Iy = gx[2112];
                            Iz = gx[3456];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[2176] * prod_xz;
                            v_iz += ai2 * gx[3520] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += (ak2 * gx[2496] - 1 * gx[1728]) * prod_xz;
                            v_kz += (ak2 * gx[3840] - 1 * gx[3072]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += (aj2 * (gx[2176] - yjyi * Iy) - 1 * gx[1920]) * prod_xz;
                            v_jz += aj2 * (gx[3520] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[7*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+1] * density_auxvec[k0+4];
                            }
                            Ix = gx[0];
                            Iy = gx[1984];
                            Iz = gx[3648];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[2048] - 1 * gx[1920]) * prod_xz;
                            v_iz += ai2 * gx[3712] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[2368] - 1 * gx[1600]) * prod_xz;
                            v_kz += (ak2 * gx[4032] - 1 * gx[3264]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[2048] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[3712] - zjzi * Iz) - 1 * gx[3456]) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+5];
                            }
                            Ix = gx[192];
                            Iy = gx[1536];
                            Iz = gx[3904];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[256] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[3968] - 1 * gx[3840]) * prod_xy;
                            v_kx += ak2 * gx[576] * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[4288] - 2 * gx[3520]) * prod_xy;
                            v_jx += (aj2 * (gx[256] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[3968] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[6*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[2*nao+0] * density_auxvec[k0+5];
                            }
                            Ix = gx[64];
                            Iy = gx[1536];
                            Iz = gx[4032];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[4096] * prod_xy;
                            v_kx += ak2 * gx[448] * prod_yz;
                            v_ky += ak2 * gx[1920] * prod_xz;
                            v_kz += (ak2 * gx[4416] - 2 * gx[3648]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += (aj2 * (gx[4096] - zjzi * Iz) - 1 * gx[3840]) * prod_xy;
                            break;
                        }
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
void int3c2e_ip1_202(double *ejk, double *ejk_aux, double *dm, double *density_auxvec,
                    RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int thread_id = threadIdx.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nksh = ksh1 - ksh0;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    constexpr int aux_per_block = 16;
    constexpr int nsp_per_block = 4;
    constexpr int nst_per_block = 64;
    int gout_id = thread_id / 64;
    int st_id = thread_id % 64;
    int sp_id = st_id / aux_per_block;
    int aux_id = st_id - sp_id * aux_per_block;
    extern __shared__ double shared_memory[];
    double *Rpq = shared_memory + st_id;
    double *gx = shared_memory + 192 + st_id;
    double *rw = shared_memory + 3264 + st_id;
    double *rjri = shared_memory + 3264 + nroots * 128 + sp_id;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
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
        if (gout_id == 0 && aux_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
            rjri[3*nsp_per_block] = rr_ij;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh;
            if (kidx < ksh1) {
                ksh = kidx;
            } else {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }
            double v_kx = 0;
            double v_ky = 0;
            double v_kz = 0;
            double s0, s1, s2;
            double dm_ijk;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
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
                __syncthreads();
                double xij = rjri[0*nsp_per_block] * aj_aij + ri[0];
                double yij = rjri[1*nsp_per_block] * aj_aij + ri[1];
                double zij = rjri[2*nsp_per_block] * aj_aij + ri[2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                if (gout_id == 0) {
                    double cijk = PI_FAC * ci[ip] * cj[jp] * ck[kp];
                    if (ish == jsh) {
                        cijk *= .5;
                    } else if (ish < jsh) {
                        cijk = 0;
                    }
                    double fac = cijk / (aij*ak*sqrt(aij+ak));
                    double theta_ij = ai * aj_aij;
                    double Kab = theta_ij * rjri[3*nsp_per_block];
                    gx[1024] = fac * exp(-Kab);
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta = aij * ak / (aij + ak);
                rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, 4);
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    double xjxi = rjri[0*nsp_per_block];  
                    double yjyi = rjri[1*nsp_per_block];  
                    double zjzi = rjri[2*nsp_per_block];  
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_ak = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/ak * (1 - rt_ak);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gx[2048] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rjri[n * nsp_per_block];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[192] = s2;
                        double cpx = rt_ak * Rpq[n * 64];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[256] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        _gx[512] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        _gx[768] = s2;
                        s0 = _gx[64];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[320] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 1 * b00 * _gx[256];
                        _gx[576] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 1 * b00 * _gx[512];
                        _gx[832] = s2;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[64];
                        _gx[384] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 2 * b00 * _gx[320];
                        _gx[640] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 2 * b00 * _gx[576];
                        _gx[896] = s2;
                        s0 = _gx[192];
                        s1 = cpx * s0;
                        s1 += 3 * b00 * _gx[128];
                        _gx[448] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 3 * b00 * _gx[384];
                        _gx[704] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 3 * b00 * _gx[640];
                        _gx[960] = s2;
                    }
                    __syncthreads();
                    if (pair_ij < shl_pair1 && kidx < ksh1) {
                        switch (gout_id) {
                        case 0:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+0];
                            }
                            Ix = gx[640];
                            Iy = gx[1024];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += (ak2 * gx[896] - 2 * gx[384]) * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[704] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1088];
                            Iz = gx[2112];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1152] - 1 * gx[1024]) * prod_xz;
                            v_iz += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 2 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[1344] * prod_xz;
                            v_kz += ak2 * gx[2368] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1152] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2176] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+1];
                            }
                            Ix = gx[320];
                            Iy = gx[1280];
                            Iz = gx[2112];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += ai2 * gx[1344] * prod_xz;
                            v_iz += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xy;
                            v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                            v_ky += (ak2 * gx[1536] - 1 * gx[1024]) * prod_xz;
                            v_kz += ak2 * gx[2368] * prod_xy;
                            v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1344] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2176] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+2];
                            }
                            Ix = gx[384];
                            Iy = gx[1024];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += (ak2 * gx[640] - 1 * gx[128]) * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += (ak2 * gx[2560] - 1 * gx[2048]) * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+2];
                            }
                            Ix = gx[256];
                            Iy = gx[1088];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1152] - 1 * gx[1024]) * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1344] * prod_xz;
                            v_kz += (ak2 * gx[2624] - 1 * gx[2112]) * prod_xy;
                            v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1152] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+3];
                            }
                            Ix = gx[64];
                            Iy = gx[1536];
                            Iz = gx[2112];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xy;
                            v_kx += ak2 * gx[320] * prod_yz;
                            v_ky += (ak2 * gx[1792] - 2 * gx[1280]) * prod_xz;
                            v_kz += ak2 * gx[2368] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2176] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+4];
                            }
                            Ix = gx[128];
                            Iy = gx[1280];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[1344] * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[1536] - 1 * gx[1024]) * prod_xz;
                            v_kz += (ak2 * gx[2560] - 1 * gx[2048]) * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1344] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+4];
                            }
                            Ix = gx[0];
                            Iy = gx[1344];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1408] - 1 * gx[1280]) * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += (ak2 * gx[1600] - 1 * gx[1088]) * prod_xz;
                            v_kz += (ak2 * gx[2624] - 1 * gx[2112]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1408] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+5];
                            }
                            Ix = gx[64];
                            Iy = gx[1024];
                            Iz = gx[2624];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += (ai2 * gx[2688] - 1 * gx[2560]) * prod_xy;
                            v_kx += ak2 * gx[320] * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += (ak2 * gx[2880] - 2 * gx[2368]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2688] - zjzi * Iz) * prod_xy;
                            break;
                        case 1:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1088];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += (ai2 * gx[1152] - 1 * gx[1024]) * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += (ak2 * gx[832] - 2 * gx[320]) * prod_yz;
                            v_ky += ak2 * gx[1344] * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1152] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1024];
                            Iz = gx[2176];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += (ai2 * gx[2240] - 2 * gx[2112]) * prod_xy;
                            v_kx += (ak2 * gx[768] - 2 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += ak2 * gx[2432] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2240] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+1];
                            }
                            Ix = gx[256];
                            Iy = gx[1408];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1472] - 2 * gx[1344]) * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                            v_ky += (ak2 * gx[1664] - 1 * gx[1152]) * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1472] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+2];
                            }
                            Ix = gx[320];
                            Iy = gx[1088];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += (ai2 * gx[1152] - 1 * gx[1024]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[1344] * prod_xz;
                            v_kz += (ak2 * gx[2560] - 1 * gx[2048]) * prod_xy;
                            v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1152] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+2];
                            }
                            Ix = gx[256];
                            Iy = gx[1024];
                            Iz = gx[2432];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += (ai2 * gx[2496] - 2 * gx[2368]) * prod_xy;
                            v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += (ak2 * gx[2688] - 1 * gx[2176]) * prod_xy;
                            v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2496] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+3];
                            }
                            Ix = gx[0];
                            Iy = gx[1664];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1728] - 2 * gx[1600]) * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += (ak2 * gx[1920] - 2 * gx[1408]) * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1728] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+4];
                            }
                            Ix = gx[64];
                            Iy = gx[1344];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[1408] - 1 * gx[1280]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += ak2 * gx[320] * prod_yz;
                            v_ky += (ak2 * gx[1600] - 1 * gx[1088]) * prod_xz;
                            v_kz += (ak2 * gx[2560] - 1 * gx[2048]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1408] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+4];
                            }
                            Ix = gx[0];
                            Iy = gx[1280];
                            Iz = gx[2432];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1344] * prod_xz;
                            v_iz += (ai2 * gx[2496] - 2 * gx[2368]) * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += (ak2 * gx[1536] - 1 * gx[1024]) * prod_xz;
                            v_kz += (ak2 * gx[2688] - 1 * gx[2176]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1344] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2496] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+5];
                            }
                            Ix = gx[0];
                            Iy = gx[1152];
                            Iz = gx[2560];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1216] - 2 * gx[1088]) * prod_xz;
                            v_iz += ai2 * gx[2624] * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += ak2 * gx[1408] * prod_xz;
                            v_kz += (ak2 * gx[2816] - 2 * gx[2304]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2624] - zjzi * Iz) * prod_xy;
                            break;
                        case 2:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+0];
                            }
                            Ix = gx[576];
                            Iy = gx[1024];
                            Iz = gx[2112];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xy;
                            v_kx += (ak2 * gx[832] - 2 * gx[320]) * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += ak2 * gx[2368] * prod_xy;
                            v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2176] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+1];
                            }
                            Ix = gx[384];
                            Iy = gx[1280];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                            v_iy += ai2 * gx[1344] * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += (ak2 * gx[640] - 1 * gx[128]) * prod_yz;
                            v_ky += (ak2 * gx[1536] - 1 * gx[1024]) * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1344] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+1];
                            }
                            Ix = gx[256];
                            Iy = gx[1344];
                            Iz = gx[2112];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1408] - 1 * gx[1280]) * prod_xz;
                            v_iz += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xy;
                            v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                            v_ky += (ak2 * gx[1600] - 1 * gx[1088]) * prod_xz;
                            v_kz += ak2 * gx[2368] * prod_xy;
                            v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1408] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2176] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+2];
                            }
                            Ix = gx[320];
                            Iy = gx[1024];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += (ak2 * gx[2624] - 1 * gx[2112]) * prod_xy;
                            v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+3];
                            }
                            Ix = gx[128];
                            Iy = gx[1536];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += (ak2 * gx[1792] - 2 * gx[1280]) * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+3];
                            }
                            Ix = gx[0];
                            Iy = gx[1600];
                            Iz = gx[2112];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += (ai2 * gx[2176] - 1 * gx[2048]) * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += (ak2 * gx[1856] - 2 * gx[1344]) * prod_xz;
                            v_kz += ak2 * gx[2368] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2176] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[2*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+2] * density_auxvec[k0+4];
                            }
                            Ix = gx[64];
                            Iy = gx[1280];
                            Iz = gx[2368];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += ai2 * gx[1344] * prod_xz;
                            v_iz += (ai2 * gx[2432] - 1 * gx[2304]) * prod_xy;
                            v_kx += ak2 * gx[320] * prod_yz;
                            v_ky += (ak2 * gx[1536] - 1 * gx[1024]) * prod_xz;
                            v_kz += (ak2 * gx[2624] - 1 * gx[2112]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1344] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2432] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[0*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+0] * density_auxvec[k0+5];
                            }
                            Ix = gx[128];
                            Iy = gx[1024];
                            Iz = gx[2560];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += ai2 * gx[2624] * prod_xy;
                            v_kx += ak2 * gx[384] * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += (ak2 * gx[2816] - 2 * gx[2304]) * prod_xy;
                            v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2624] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[4*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+4] * density_auxvec[k0+5];
                            }
                            Ix = gx[0];
                            Iy = gx[1088];
                            Iz = gx[2624];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1152] - 1 * gx[1024]) * prod_xz;
                            v_iz += (ai2 * gx[2688] - 1 * gx[2560]) * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += ak2 * gx[1344] * prod_xz;
                            v_kz += (ak2 * gx[2880] - 2 * gx[2368]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1152] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2688] - zjzi * Iz) * prod_xy;
                            break;
                        case 3:
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 0*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+0];
                            }
                            Ix = gx[512];
                            Iy = gx[1152];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[576] * prod_yz;
                            v_iy += (ai2 * gx[1216] - 2 * gx[1088]) * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += (ak2 * gx[768] - 2 * gx[256]) * prod_yz;
                            v_ky += ak2 * gx[1408] * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+1];
                            }
                            Ix = gx[320];
                            Iy = gx[1344];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                            v_iy += (ai2 * gx[1408] - 1 * gx[1280]) * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                            v_ky += (ak2 * gx[1600] - 1 * gx[1088]) * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1408] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 1*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+1];
                            }
                            Ix = gx[256];
                            Iy = gx[1280];
                            Iz = gx[2176];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += ai2 * gx[1344] * prod_xz;
                            v_iz += (ai2 * gx[2240] - 2 * gx[2112]) * prod_xy;
                            v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                            v_ky += (ak2 * gx[1536] - 1 * gx[1024]) * prod_xz;
                            v_kz += ak2 * gx[2432] * prod_xy;
                            v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1344] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2240] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 2*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+2];
                            }
                            Ix = gx[256];
                            Iy = gx[1152];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[320] * prod_yz;
                            v_iy += (ai2 * gx[1216] - 2 * gx[1088]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                            v_ky += ak2 * gx[1408] * prod_xz;
                            v_kz += (ak2 * gx[2560] - 1 * gx[2048]) * prod_xy;
                            v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1216] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+3];
                            }
                            Ix = gx[64];
                            Iy = gx[1600];
                            Iz = gx[2048];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[1664] - 1 * gx[1536]) * prod_xz;
                            v_iz += ai2 * gx[2112] * prod_xy;
                            v_kx += ak2 * gx[320] * prod_yz;
                            v_ky += (ak2 * gx[1856] - 2 * gx[1344]) * prod_xz;
                            v_kz += ak2 * gx[2304] * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1664] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2112] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 3*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+3];
                            }
                            Ix = gx[0];
                            Iy = gx[1536];
                            Iz = gx[2176];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1600] * prod_xz;
                            v_iz += (ai2 * gx[2240] - 2 * gx[2112]) * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += (ak2 * gx[1792] - 2 * gx[1280]) * prod_xz;
                            v_kz += ak2 * gx[2432] * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1600] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2240] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[3*naux + 4*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+3] * density_auxvec[k0+4];
                            }
                            Ix = gx[0];
                            Iy = gx[1408];
                            Iz = gx[2304];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += (ai2 * gx[1472] - 2 * gx[1344]) * prod_xz;
                            v_iz += ai2 * gx[2368] * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += (ak2 * gx[1664] - 1 * gx[1152]) * prod_xz;
                            v_kz += (ak2 * gx[2560] - 1 * gx[2048]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1472] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2368] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[1*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+1] * density_auxvec[k0+5];
                            }
                            Ix = gx[64];
                            Iy = gx[1088];
                            Iz = gx[2560];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                            v_iy += (ai2 * gx[1152] - 1 * gx[1024]) * prod_xz;
                            v_iz += ai2 * gx[2624] * prod_xy;
                            v_kx += ak2 * gx[320] * prod_yz;
                            v_ky += ak2 * gx[1344] * prod_xz;
                            v_kz += (ak2 * gx[2816] - 2 * gx[2304]) * prod_xy;
                            v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1152] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2624] - zjzi * Iz) * prod_xy;
                            if (density_auxvec == NULL) {
                                dm_ijk = dm_tensor[5*naux + 5*nksh];
                            } else {
                                dm_ijk = dm_tensor[0*nao+5] * density_auxvec[k0+5];
                            }
                            Ix = gx[0];
                            Iy = gx[1024];
                            Iz = gx[2688];
                            prod_xy = Ix * Iy * dm_ijk;
                            prod_xz = Ix * Iz * dm_ijk;
                            prod_yz = Iy * Iz * dm_ijk;
                            v_ix += ai2 * gx[64] * prod_yz;
                            v_iy += ai2 * gx[1088] * prod_xz;
                            v_iz += (ai2 * gx[2752] - 2 * gx[2624]) * prod_xy;
                            v_kx += ak2 * gx[256] * prod_yz;
                            v_ky += ak2 * gx[1280] * prod_xz;
                            v_kz += (ak2 * gx[2944] - 2 * gx[2432]) * prod_xy;
                            v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                            v_jy += aj2 * (gx[1088] - yjyi * Iy) * prod_xz;
                            v_jz += aj2 * (gx[2752] - zjzi * Iz) * prod_xy;
                            break;
                        }
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
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int aux_offset, int naux, int nao)
{
    int kij_type = lk*25 + li*5 + lj;
    switch (kij_type) {
    case 0: // li=0 lj=0 lk=0
        int3c2e_ip1_000(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 5: // li=1 lj=0 lk=0
        int3c2e_ip1_100(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 6: // li=1 lj=1 lk=0
        int3c2e_ip1_110(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 10: // li=2 lj=0 lk=0
        int3c2e_ip1_200(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 11: // li=2 lj=1 lk=0
        int3c2e_ip1_210(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 12: // li=2 lj=2 lk=0
        int3c2e_ip1_220(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 25: // li=0 lj=0 lk=1
        int3c2e_ip1_001(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 30: // li=1 lj=0 lk=1
        int3c2e_ip1_101(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 31: // li=1 lj=1 lk=1
        int3c2e_ip1_111(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 35: // li=2 lj=0 lk=1
        int3c2e_ip1_201(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 36: // li=2 lj=1 lk=1
        int3c2e_ip1_211(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 50: // li=0 lj=0 lk=2
        int3c2e_ip1_002(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 55: // li=1 lj=0 lk=2
        int3c2e_ip1_102(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 56: // li=1 lj=1 lk=2
        int3c2e_ip1_112(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    case 60: // li=2 lj=0 lk=2
        int3c2e_ip1_202(ejk, ejk_aux, dm, density_auxvec, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, aux_offset, naux, nao); break;
    default: return 0;
    }
    return 1;
}
