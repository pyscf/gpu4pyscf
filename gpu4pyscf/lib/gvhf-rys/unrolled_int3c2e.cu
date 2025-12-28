#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"


__device__ inline
void int3c2e_000(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 1;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[1];
        for (int n = 0; n < 1; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                gout[0] += 1 * fac1 * wt;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 1; ++n) {
                int k = n / 1;
                int ij = n - 1 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 1;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 1; ++n) {
                int k = n / 1;
                int ij = n - 1 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_100(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 1;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[3];
        for (int n = 0; n < 3; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                gout[0] += trr_10x * fac1 * wt;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += 1 * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += 1 * fac1 * trr_10z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 3; ++n) {
                int k = n / 3;
                int ij = n - 3 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 1;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 3; ++n) {
                int k = n / 3;
                int ij = n - 3 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_110(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[9];
        for (int n = 0; n < 9; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                gout[0] += hrr_110x * fac1 * wt;
                double hrr_010x = trr_10x - xjxi * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += hrr_010x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += hrr_010x * fac1 * trr_10z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout[3] += trr_10x * hrr_010y * wt;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout[4] += 1 * hrr_110y * wt;
                gout[5] += 1 * hrr_010y * trr_10z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout[6] += trr_10x * fac1 * hrr_010z;
                gout[7] += 1 * trr_10y * hrr_010z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout[8] += 1 * fac1 * hrr_110z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 9; ++n) {
                int k = n / 9;
                int ij = n - 9 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 1;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 9; ++n) {
                int k = n / 9;
                int ij = n - 9 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_200(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[6];
        for (int n = 0; n < 6; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                gout[0] += trr_20x * fac1 * wt;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += trr_10x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += trr_10x * fac1 * trr_10z;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout[3] += 1 * trr_20y * wt;
                gout[4] += 1 * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout[5] += 1 * fac1 * trr_20z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 6; ++n) {
                int k = n / 6;
                int ij = n - 6 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 1;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 6; ++n) {
                int k = n / 6;
                int ij = n - 6 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_210(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[18];
        for (int n = 0; n < 18; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                double hrr_210x = trr_30x - xjxi * trr_20x;
                gout[0] += hrr_210x * fac1 * wt;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += hrr_110x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += hrr_110x * fac1 * trr_10z;
                double hrr_010x = trr_10x - xjxi * 1;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout[3] += hrr_010x * trr_20y * wt;
                gout[4] += hrr_010x * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout[5] += hrr_010x * fac1 * trr_20z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout[6] += trr_20x * hrr_010y * wt;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout[7] += trr_10x * hrr_110y * wt;
                gout[8] += trr_10x * hrr_010y * trr_10z;
                double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                double hrr_210y = trr_30y - yjyi * trr_20y;
                gout[9] += 1 * hrr_210y * wt;
                gout[10] += 1 * hrr_110y * trr_10z;
                gout[11] += 1 * hrr_010y * trr_20z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout[12] += trr_20x * fac1 * hrr_010z;
                gout[13] += trr_10x * trr_10y * hrr_010z;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout[14] += trr_10x * fac1 * hrr_110z;
                gout[15] += 1 * trr_20y * hrr_010z;
                gout[16] += 1 * trr_10y * hrr_110z;
                double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                double hrr_210z = trr_30z - zjzi * trr_20z;
                gout[17] += 1 * fac1 * hrr_210z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 18; ++n) {
                int k = n / 18;
                int ij = n - 18 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 1;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 18; ++n) {
                int k = n / 18;
                int ij = n - 18 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_220(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int thread_id = threadIdx.x;
    int st_id = thread_id % 128;
    int gout_id = thread_id / 128;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 256;
    double *gy = gx + 1152;
    double *gz = gx + 2304;
    double *Rpq = gx + 3456;
    double *rjri = gx + 3840;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 128) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[128] = yjyi;
            rjri[256] = zjzi;
            rjri[384] = rr_ij;
        }
        double gout[18];
        for (int n = 0; n < 18; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        double s0, s1, s2;
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
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[128] * aj_aij + ri[1];
            double zij = rjri[256] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[384];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[128] = ypq;
                Rpq[256] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, 128, gout_id, 2);
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*256];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                for (int n = gout_id; n < 3; n += 2) {
                    if (n == 2) {
                        gz[0] = rw[irys*256+128];
                    }
                    double *_gx = gx + n * 1152;
                    double xjxi = rjri[n * 128];
                    double Rpa = xjxi * aj_aij;
                    double c0x = Rpa - rt_aij * Rpq[n * 128];
                    s0 = _gx[0];
                    s1 = c0x * s0;
                    _gx[128] = s1;
                    s2 = c0x * s1 + 1 * b10 * s0;
                    _gx[256] = s2;
                    s0 = s1;
                    s1 = s2;
                    s2 = c0x * s1 + 2 * b10 * s0;
                    _gx[384] = s2;
                    s0 = s1;
                    s1 = s2;
                    s2 = c0x * s1 + 3 * b10 * s0;
                    _gx[512] = s2;
                    s1 = _gx[512];
                    s0 = _gx[384];
                    _gx[768] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[256];
                    _gx[640] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[128];
                    _gx[512] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[0];
                    _gx[384] = s1 - xjxi * s0;
                    s1 = _gx[768];
                    s0 = _gx[640];
                    _gx[1024] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[512];
                    _gx[896] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[384];
                    _gx[768] = s1 - xjxi * s0;
                }
                __syncthreads();
                switch (gout_id) {
                case 0:
                gout[0] += gx[1024] * gy[0] * gz[0];
                gout[1] += gx[896] * gy[0] * gz[128];
                gout[2] += gx[768] * gy[128] * gz[128];
                gout[3] += gx[640] * gy[384] * gz[0];
                gout[4] += gx[512] * gy[384] * gz[128];
                gout[5] += gx[384] * gy[512] * gz[128];
                gout[6] += gx[640] * gy[0] * gz[384];
                gout[7] += gx[512] * gy[0] * gz[512];
                gout[8] += gx[384] * gy[128] * gz[512];
                gout[9] += gx[256] * gy[768] * gz[0];
                gout[10] += gx[128] * gy[768] * gz[128];
                gout[11] += gx[0] * gy[896] * gz[128];
                gout[12] += gx[256] * gy[384] * gz[384];
                gout[13] += gx[128] * gy[384] * gz[512];
                gout[14] += gx[0] * gy[512] * gz[512];
                gout[15] += gx[256] * gy[0] * gz[768];
                gout[16] += gx[128] * gy[0] * gz[896];
                gout[17] += gx[0] * gy[128] * gz[896];
                break;
                case 1:
                gout[0] += gx[896] * gy[128] * gz[0];
                gout[1] += gx[768] * gy[256] * gz[0];
                gout[2] += gx[768] * gy[0] * gz[256];
                gout[3] += gx[512] * gy[512] * gz[0];
                gout[4] += gx[384] * gy[640] * gz[0];
                gout[5] += gx[384] * gy[384] * gz[256];
                gout[6] += gx[512] * gy[128] * gz[384];
                gout[7] += gx[384] * gy[256] * gz[384];
                gout[8] += gx[384] * gy[0] * gz[640];
                gout[9] += gx[128] * gy[896] * gz[0];
                gout[10] += gx[0] * gy[1024] * gz[0];
                gout[11] += gx[0] * gy[768] * gz[256];
                gout[12] += gx[128] * gy[512] * gz[384];
                gout[13] += gx[0] * gy[640] * gz[384];
                gout[14] += gx[0] * gy[384] * gz[640];
                gout[15] += gx[128] * gy[128] * gz[768];
                gout[16] += gx[0] * gy[256] * gz[768];
                gout[17] += gx[0] * gy[0] * gz[1024];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            if (reorder_aux) {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 18; ++n) {
                    int ijk = n*2+gout_id;
                    if (ijk >= 36) break;
                    int ij = ijk / 1;
                    int k  = ijk - 1 * ij;
                    j3c_tensor[ij*naux + k*nksh] = gout[n];
                }
            } else {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 1;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 18; ++n) {
                    int ijk = n*2+gout_id;
                    if (ijk >= 36) break;
                    int ij = ijk / 1;
                    int k  = ijk - 1 * ij;
                    j3c_tensor[ij*naux + k] = gout[n];
                }
            }
        }
    }
}

__device__ inline
void int3c2e_001(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 1;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[3];
        for (int n = 0; n < 3; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double trr_01x = cpx * 1;
                gout[0] += trr_01x * fac1 * wt;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout[1] += 1 * trr_01y * wt;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout[2] += 1 * fac1 * trr_01z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 3; ++n) {
                int k = n / 1;
                int ij = n - 1 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 3;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 3; ++n) {
                int k = n / 1;
                int ij = n - 1 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_101(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[9];
        for (int n = 0; n < 9; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                gout[0] += trr_11x * fac1 * wt;
                double trr_01x = cpx * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += trr_01x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += trr_01x * fac1 * trr_10z;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout[3] += trr_10x * trr_01y * wt;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout[4] += 1 * trr_11y * wt;
                gout[5] += 1 * trr_01y * trr_10z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout[6] += trr_10x * fac1 * trr_01z;
                gout[7] += 1 * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout[8] += 1 * fac1 * trr_11z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 9; ++n) {
                int k = n / 3;
                int ij = n - 3 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 3;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 9; ++n) {
                int k = n / 3;
                int ij = n - 3 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_111(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[27];
        for (int n = 0; n < 27; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double hrr_111x = trr_21x - xjxi * trr_11x;
                gout[0] += hrr_111x * fac1 * wt;
                double trr_01x = cpx * 1;
                double hrr_011x = trr_11x - xjxi * trr_01x;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += hrr_011x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += hrr_011x * fac1 * trr_10z;
                double hrr_010y = trr_10y - yjyi * fac1;
                gout[3] += trr_11x * hrr_010y * wt;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                double hrr_110y = trr_20y - yjyi * trr_10y;
                gout[4] += trr_01x * hrr_110y * wt;
                gout[5] += trr_01x * hrr_010y * trr_10z;
                double hrr_010z = trr_10z - zjzi * wt;
                gout[6] += trr_11x * fac1 * hrr_010z;
                gout[7] += trr_01x * trr_10y * hrr_010z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                double hrr_110z = trr_20z - zjzi * trr_10z;
                gout[8] += trr_01x * fac1 * hrr_110z;
                double hrr_110x = trr_20x - xjxi * trr_10x;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout[9] += hrr_110x * trr_01y * wt;
                double hrr_010x = trr_10x - xjxi * 1;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout[10] += hrr_010x * trr_11y * wt;
                gout[11] += hrr_010x * trr_01y * trr_10z;
                double hrr_011y = trr_11y - yjyi * trr_01y;
                gout[12] += trr_10x * hrr_011y * wt;
                double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                double hrr_111y = trr_21y - yjyi * trr_11y;
                gout[13] += 1 * hrr_111y * wt;
                gout[14] += 1 * hrr_011y * trr_10z;
                gout[15] += trr_10x * trr_01y * hrr_010z;
                gout[16] += 1 * trr_11y * hrr_010z;
                gout[17] += 1 * trr_01y * hrr_110z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout[18] += hrr_110x * fac1 * trr_01z;
                gout[19] += hrr_010x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout[20] += hrr_010x * fac1 * trr_11z;
                gout[21] += trr_10x * hrr_010y * trr_01z;
                gout[22] += 1 * hrr_110y * trr_01z;
                gout[23] += 1 * hrr_010y * trr_11z;
                double hrr_011z = trr_11z - zjzi * trr_01z;
                gout[24] += trr_10x * fac1 * hrr_011z;
                gout[25] += 1 * trr_10y * hrr_011z;
                double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                double hrr_111z = trr_21z - zjzi * trr_11z;
                gout[26] += 1 * fac1 * hrr_111z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 27; ++n) {
                int k = n / 9;
                int ij = n - 9 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 3;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 27; ++n) {
                int k = n / 9;
                int ij = n - 9 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_201(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[18];
        for (int n = 0; n < 18; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_20x = c0x * trr_10x + 1*b10 * 1;
                double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                gout[0] += trr_21x * fac1 * wt;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += trr_11x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += trr_11x * fac1 * trr_10z;
                double trr_01x = cpx * 1;
                double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                gout[3] += trr_01x * trr_20y * wt;
                gout[4] += trr_01x * trr_10y * trr_10z;
                double trr_20z = c0z * trr_10z + 1*b10 * wt;
                gout[5] += trr_01x * fac1 * trr_20z;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout[6] += trr_20x * trr_01y * wt;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout[7] += trr_10x * trr_11y * wt;
                gout[8] += trr_10x * trr_01y * trr_10z;
                double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                gout[9] += 1 * trr_21y * wt;
                gout[10] += 1 * trr_11y * trr_10z;
                gout[11] += 1 * trr_01y * trr_20z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout[12] += trr_20x * fac1 * trr_01z;
                gout[13] += trr_10x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout[14] += trr_10x * fac1 * trr_11z;
                gout[15] += 1 * trr_20y * trr_01z;
                gout[16] += 1 * trr_10y * trr_11z;
                double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                gout[17] += 1 * fac1 * trr_21z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 18; ++n) {
                int k = n / 6;
                int ij = n - 6 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 3;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 18; ++n) {
                int k = n / 6;
                int ij = n - 6 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_211(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int thread_id = threadIdx.x;
    int st_id = thread_id % 64;
    int gout_id = thread_id / 64;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 128;
    double *gy = gx + 768;
    double *gz = gx + 1536;
    double *Rpq = gx + 2304;
    double *rjri = gx + 2496;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 64) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[64] = yjyi;
            rjri[128] = zjzi;
            rjri[192] = rr_ij;
        }
        double gout[14];
        for (int n = 0; n < 14; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        double s0, s1, s2;
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
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[64] * aj_aij + ri[1];
            double zij = rjri[128] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[192];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, 64, gout_id, 4);
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*128];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                for (int n = gout_id; n < 3; n += 4) {
                    if (n == 2) {
                        gz[0] = rw[irys*128+64];
                    }
                    double *_gx = gx + n * 768;
                    double xjxi = rjri[n * 64];
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
                    s0 = _gx[64];
                    s1 = cpx * s0;
                    s1 += 1 * b00 * _gx[0];
                    _gx[448] = s1;
                    s0 = _gx[128];
                    s1 = cpx * s0;
                    s1 += 2 * b00 * _gx[64];
                    _gx[512] = s1;
                    s0 = _gx[192];
                    s1 = cpx * s0;
                    s1 += 3 * b00 * _gx[128];
                    _gx[576] = s1;
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
                }
                __syncthreads();
                switch (gout_id) {
                case 0:
                gout[0] += gx[704] * gy[0] * gz[0];
                gout[1] += gx[256] * gy[448] * gz[0];
                gout[2] += gx[256] * gy[0] * gz[448];
                gout[3] += gx[576] * gy[64] * gz[64];
                gout[4] += gx[192] * gy[384] * gz[128];
                gout[5] += gx[128] * gy[192] * gz[384];
                gout[6] += gx[448] * gy[192] * gz[64];
                gout[7] += gx[0] * gy[704] * gz[0];
                gout[8] += gx[0] * gy[256] * gz[448];
                gout[9] += gx[512] * gy[0] * gz[192];
                gout[10] += gx[64] * gy[448] * gz[192];
                gout[11] += gx[64] * gy[0] * gz[640];
                gout[12] += gx[384] * gy[64] * gz[256];
                gout[13] += gx[0] * gy[384] * gz[320];
                break;
                case 1:
                gout[0] += gx[320] * gy[384] * gz[0];
                gout[1] += gx[256] * gy[64] * gz[384];
                gout[2] += gx[576] * gy[128] * gz[0];
                gout[3] += gx[192] * gy[448] * gz[64];
                gout[4] += gx[192] * gy[0] * gz[512];
                gout[5] += gx[448] * gy[256] * gz[0];
                gout[6] += gx[64] * gy[576] * gz[64];
                gout[7] += gx[0] * gy[320] * gz[384];
                gout[8] += gx[384] * gy[192] * gz[128];
                gout[9] += gx[128] * gy[384] * gz[192];
                gout[10] += gx[64] * gy[64] * gz[576];
                gout[11] += gx[384] * gy[128] * gz[192];
                gout[12] += gx[0] * gy[448] * gz[256];
                gout[13] += gx[0] * gy[0] * gz[704];
                break;
                case 2:
                gout[0] += gx[320] * gy[0] * gz[384];
                gout[1] += gx[640] * gy[0] * gz[64];
                gout[2] += gx[192] * gy[512] * gz[0];
                gout[3] += gx[192] * gy[64] * gz[448];
                gout[4] += gx[512] * gy[192] * gz[0];
                gout[5] += gx[64] * gy[640] * gz[0];
                gout[6] += gx[64] * gy[192] * gz[448];
                gout[7] += gx[384] * gy[256] * gz[64];
                gout[8] += gx[0] * gy[576] * gz[128];
                gout[9] += gx[128] * gy[0] * gz[576];
                gout[10] += gx[448] * gy[0] * gz[256];
                gout[11] += gx[0] * gy[512] * gz[192];
                gout[12] += gx[0] * gy[64] * gz[640];
                break;
                case 3:
                gout[0] += gx[640] * gy[64] * gz[0];
                gout[1] += gx[256] * gy[384] * gz[64];
                gout[2] += gx[192] * gy[128] * gz[384];
                gout[3] += gx[576] * gy[0] * gz[128];
                gout[4] += gx[128] * gy[576] * gz[0];
                gout[5] += gx[64] * gy[256] * gz[384];
                gout[6] += gx[384] * gy[320] * gz[0];
                gout[7] += gx[0] * gy[640] * gz[64];
                gout[8] += gx[0] * gy[192] * gz[512];
                gout[9] += gx[448] * gy[64] * gz[192];
                gout[10] += gx[64] * gy[384] * gz[256];
                gout[11] += gx[0] * gy[128] * gz[576];
                gout[12] += gx[384] * gy[0] * gz[320];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            if (reorder_aux) {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 14; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 54) break;
                    int ij = ijk / 3;
                    int k  = ijk - 3 * ij;
                    j3c_tensor[ij*naux + k*nksh] = gout[n];
                }
            } else {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 3;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 14; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 54) break;
                    int ij = ijk / 3;
                    int k  = ijk - 3 * ij;
                    j3c_tensor[ij*naux + k] = gout[n];
                }
            }
        }
    }
}

__device__ inline
void int3c2e_221(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int thread_id = threadIdx.x;
    int st_id = thread_id % 64;
    int gout_id = thread_id / 64;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 128;
    double *gy = gx + 1152;
    double *gz = gx + 2304;
    double *Rpq = gx + 3456;
    double *rjri = gx + 3648;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 64) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[64] = yjyi;
            rjri[128] = zjzi;
            rjri[192] = rr_ij;
        }
        double gout[27];
        for (int n = 0; n < 27; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        double s0, s1, s2;
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
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[64] * aj_aij + ri[1];
            double zij = rjri[128] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[192];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, 64, gout_id, 4);
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*128];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                for (int n = gout_id; n < 3; n += 4) {
                    if (n == 2) {
                        gz[0] = rw[irys*128+64];
                    }
                    double *_gx = gx + n * 1152;
                    double xjxi = rjri[n * 64];
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
                    _gx[576] = s1;
                    s0 = _gx[64];
                    s1 = cpx * s0;
                    s1 += 1 * b00 * _gx[0];
                    _gx[640] = s1;
                    s0 = _gx[128];
                    s1 = cpx * s0;
                    s1 += 2 * b00 * _gx[64];
                    _gx[704] = s1;
                    s0 = _gx[192];
                    s1 = cpx * s0;
                    s1 += 3 * b00 * _gx[128];
                    _gx[768] = s1;
                    s0 = _gx[256];
                    s1 = cpx * s0;
                    s1 += 4 * b00 * _gx[192];
                    _gx[832] = s1;
                    s1 = _gx[256];
                    s0 = _gx[192];
                    _gx[384] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[128];
                    _gx[320] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[64];
                    _gx[256] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[0];
                    _gx[192] = s1 - xjxi * s0;
                    s1 = _gx[384];
                    s0 = _gx[320];
                    _gx[512] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[256];
                    _gx[448] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[192];
                    _gx[384] = s1 - xjxi * s0;
                    s1 = _gx[832];
                    s0 = _gx[768];
                    _gx[960] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[704];
                    _gx[896] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[640];
                    _gx[832] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[576];
                    _gx[768] = s1 - xjxi * s0;
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
                switch (gout_id) {
                case 0:
                gout[0] += gx[1088] * gy[0] * gz[0];
                gout[1] += gx[448] * gy[640] * gz[0];
                gout[2] += gx[448] * gy[0] * gz[640];
                gout[3] += gx[960] * gy[64] * gz[64];
                gout[4] += gx[384] * gy[576] * gz[128];
                gout[5] += gx[320] * gy[192] * gz[576];
                gout[6] += gx[832] * gy[192] * gz[64];
                gout[7] += gx[192] * gy[896] * gz[0];
                gout[8] += gx[192] * gy[256] * gz[640];
                gout[9] += gx[896] * gy[0] * gz[192];
                gout[10] += gx[256] * gy[640] * gz[192];
                gout[11] += gx[256] * gy[0] * gz[832];
                gout[12] += gx[768] * gy[64] * gz[256];
                gout[13] += gx[192] * gy[576] * gz[320];
                gout[14] += gx[128] * gy[384] * gz[576];
                gout[15] += gx[640] * gy[384] * gz[64];
                gout[16] += gx[0] * gy[1088] * gz[0];
                gout[17] += gx[0] * gy[448] * gz[640];
                gout[18] += gx[704] * gy[192] * gz[192];
                gout[19] += gx[64] * gy[832] * gz[192];
                gout[20] += gx[64] * gy[192] * gz[832];
                gout[21] += gx[576] * gy[256] * gz[256];
                gout[22] += gx[0] * gy[768] * gz[320];
                gout[23] += gx[128] * gy[0] * gz[960];
                gout[24] += gx[640] * gy[0] * gz[448];
                gout[25] += gx[0] * gy[704] * gz[384];
                gout[26] += gx[0] * gy[64] * gz[1024];
                break;
                case 1:
                gout[0] += gx[512] * gy[576] * gz[0];
                gout[1] += gx[448] * gy[64] * gz[576];
                gout[2] += gx[960] * gy[128] * gz[0];
                gout[3] += gx[384] * gy[640] * gz[64];
                gout[4] += gx[384] * gy[0] * gz[704];
                gout[5] += gx[832] * gy[256] * gz[0];
                gout[6] += gx[256] * gy[768] * gz[64];
                gout[7] += gx[192] * gy[320] * gz[576];
                gout[8] += gx[768] * gy[192] * gz[128];
                gout[9] += gx[320] * gy[576] * gz[192];
                gout[10] += gx[256] * gy[64] * gz[768];
                gout[11] += gx[768] * gy[128] * gz[192];
                gout[12] += gx[192] * gy[640] * gz[256];
                gout[13] += gx[192] * gy[0] * gz[896];
                gout[14] += gx[640] * gy[448] * gz[0];
                gout[15] += gx[64] * gy[960] * gz[64];
                gout[16] += gx[0] * gy[512] * gz[576];
                gout[17] += gx[576] * gy[384] * gz[128];
                gout[18] += gx[128] * gy[768] * gz[192];
                gout[19] += gx[64] * gy[256] * gz[768];
                gout[20] += gx[576] * gy[320] * gz[192];
                gout[21] += gx[0] * gy[832] * gz[256];
                gout[22] += gx[0] * gy[192] * gz[896];
                gout[23] += gx[640] * gy[64] * gz[384];
                gout[24] += gx[64] * gy[576] * gz[448];
                gout[25] += gx[0] * gy[128] * gz[960];
                gout[26] += gx[576] * gy[0] * gz[512];
                break;
                case 2:
                gout[0] += gx[512] * gy[0] * gz[576];
                gout[1] += gx[1024] * gy[0] * gz[64];
                gout[2] += gx[384] * gy[704] * gz[0];
                gout[3] += gx[384] * gy[64] * gz[640];
                gout[4] += gx[896] * gy[192] * gz[0];
                gout[5] += gx[256] * gy[832] * gz[0];
                gout[6] += gx[256] * gy[192] * gz[640];
                gout[7] += gx[768] * gy[256] * gz[64];
                gout[8] += gx[192] * gy[768] * gz[128];
                gout[9] += gx[320] * gy[0] * gz[768];
                gout[10] += gx[832] * gy[0] * gz[256];
                gout[11] += gx[192] * gy[704] * gz[192];
                gout[12] += gx[192] * gy[64] * gz[832];
                gout[13] += gx[704] * gy[384] * gz[0];
                gout[14] += gx[64] * gy[1024] * gz[0];
                gout[15] += gx[64] * gy[384] * gz[640];
                gout[16] += gx[576] * gy[448] * gz[64];
                gout[17] += gx[0] * gy[960] * gz[128];
                gout[18] += gx[128] * gy[192] * gz[768];
                gout[19] += gx[640] * gy[192] * gz[256];
                gout[20] += gx[0] * gy[896] * gz[192];
                gout[21] += gx[0] * gy[256] * gz[832];
                gout[22] += gx[704] * gy[0] * gz[384];
                gout[23] += gx[64] * gy[640] * gz[384];
                gout[24] += gx[64] * gy[0] * gz[1024];
                gout[25] += gx[576] * gy[64] * gz[448];
                gout[26] += gx[0] * gy[576] * gz[512];
                break;
                case 3:
                gout[0] += gx[1024] * gy[64] * gz[0];
                gout[1] += gx[448] * gy[576] * gz[64];
                gout[2] += gx[384] * gy[128] * gz[576];
                gout[3] += gx[960] * gy[0] * gz[128];
                gout[4] += gx[320] * gy[768] * gz[0];
                gout[5] += gx[256] * gy[256] * gz[576];
                gout[6] += gx[768] * gy[320] * gz[0];
                gout[7] += gx[192] * gy[832] * gz[64];
                gout[8] += gx[192] * gy[192] * gz[704];
                gout[9] += gx[832] * gy[64] * gz[192];
                gout[10] += gx[256] * gy[576] * gz[256];
                gout[11] += gx[192] * gy[128] * gz[768];
                gout[12] += gx[768] * gy[0] * gz[320];
                gout[13] += gx[128] * gy[960] * gz[0];
                gout[14] += gx[64] * gy[448] * gz[576];
                gout[15] += gx[576] * gy[512] * gz[0];
                gout[16] += gx[0] * gy[1024] * gz[64];
                gout[17] += gx[0] * gy[384] * gz[704];
                gout[18] += gx[640] * gy[256] * gz[192];
                gout[19] += gx[64] * gy[768] * gz[256];
                gout[20] += gx[0] * gy[320] * gz[768];
                gout[21] += gx[576] * gy[192] * gz[320];
                gout[22] += gx[128] * gy[576] * gz[384];
                gout[23] += gx[64] * gy[64] * gz[960];
                gout[24] += gx[576] * gy[128] * gz[384];
                gout[25] += gx[0] * gy[640] * gz[448];
                gout[26] += gx[0] * gy[0] * gz[1088];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            if (reorder_aux) {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 27; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 108) break;
                    int ij = ijk / 3;
                    int k  = ijk - 3 * ij;
                    j3c_tensor[ij*naux + k*nksh] = gout[n];
                }
            } else {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 3;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 27; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 108) break;
                    int ij = ijk / 3;
                    int k  = ijk - 3 * ij;
                    j3c_tensor[ij*naux + k] = gout[n];
                }
            }
        }
    }
}

__device__ inline
void int3c2e_002(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[6];
        for (int n = 0; n < 6; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double rt_ak = rt_aa * aij;
                double b01 = .5/ak * (1 - rt_ak);
                double cpx = xpq*rt_ak;
                double trr_01x = cpx * 1;
                double trr_02x = cpx * trr_01x + 1*b01 * 1;
                gout[0] += trr_02x * fac1 * wt;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout[1] += trr_01x * trr_01y * wt;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout[2] += trr_01x * fac1 * trr_01z;
                double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                gout[3] += 1 * trr_02y * wt;
                gout[4] += 1 * trr_01y * trr_01z;
                double trr_02z = cpz * trr_01z + 1*b01 * wt;
                gout[5] += 1 * fac1 * trr_02z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 6; ++n) {
                int k = n / 1;
                int ij = n - 1 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 6;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 6; ++n) {
                int k = n / 1;
                int ij = n - 1 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_102(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int st_id = threadIdx.x;
    int nst_per_block = blockDim.x;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 2;
    if (omega < 0) {
        nroots *= 2;
    }
    extern __shared__ double rw_buffer[];
    double *rw = rw_buffer + st_id;
    double *rjri = rw_buffer + nst_per_block * nroots*2 + st_id;
    for (int ijk_idx = st_id; ijk_idx < nst; ijk_idx += nst_per_block) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        int ksh = ksh_in_block + ksh0;
        int pair_ij = shl_pair_in_block + shl_pair0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        rjri[0*nst_per_block] = rj[0] - ri[0];
        rjri[1*nst_per_block] = rj[1] - ri[1];
        rjri[2*nst_per_block] = rj[2] - ri[2];
        rjri[3*nst_per_block] = rr_ij;
        double gout[18];
        for (int n = 0; n < 18; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int ijp = ijkp / kprim;
            int kp = ijkp - kprim * ijp;
            int ip = ijp / jprim;
            int jp = ijp - jprim * ip;
            double ai = expi[ip];
            double aj = expj[jp];
            double ak = expk[kp];
            double aij = ai + aj;
            double cijk = ci[ip] * cj[jp] * ck[kp];
            double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rjri[3*nst_per_block];
            double fac1 = fac * exp(-Kab);
            double xij = rjri[0*nst_per_block] * aj_aij + ri[0];
            double yij = rjri[1*nst_per_block] * aj_aij + ri[1];
            double zij = rjri[2*nst_per_block] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, 0, 1);
            for (int irys = 0; irys < nroots; ++irys) {
                double wt = rw[(2*irys+1)*nst_per_block];
                double rt = rw[ 2*irys   *nst_per_block];
                double rt_aa = rt / (aij + ak);
                double b00 = .5 * rt_aa;
                double rt_ak = rt_aa * aij;
                double b01 = .5/ak * (1 - rt_ak);
                double cpx = xpq*rt_ak;
                double rt_aij = rt_aa * ak;
                double c0x = rjri[0*nst_per_block] * aj_aij - xpq*rt_aij;
                double trr_10x = c0x * 1;
                double trr_11x = cpx * trr_10x + 1*b00 * 1;
                double trr_01x = cpx * 1;
                double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                gout[0] += trr_12x * fac1 * wt;
                double trr_02x = cpx * trr_01x + 1*b01 * 1;
                double c0y = rjri[1*nst_per_block] * aj_aij - ypq*rt_aij;
                double trr_10y = c0y * fac1;
                gout[1] += trr_02x * trr_10y * wt;
                double c0z = rjri[2*nst_per_block] * aj_aij - zpq*rt_aij;
                double trr_10z = c0z * wt;
                gout[2] += trr_02x * fac1 * trr_10z;
                double cpy = ypq*rt_ak;
                double trr_01y = cpy * fac1;
                gout[3] += trr_11x * trr_01y * wt;
                double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                gout[4] += trr_01x * trr_11y * wt;
                gout[5] += trr_01x * trr_01y * trr_10z;
                double cpz = zpq*rt_ak;
                double trr_01z = cpz * wt;
                gout[6] += trr_11x * fac1 * trr_01z;
                gout[7] += trr_01x * trr_10y * trr_01z;
                double trr_11z = cpz * trr_10z + 1*b00 * wt;
                gout[8] += trr_01x * fac1 * trr_11z;
                double trr_02y = cpy * trr_01y + 1*b01 * fac1;
                gout[9] += trr_10x * trr_02y * wt;
                double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                gout[10] += 1 * trr_12y * wt;
                gout[11] += 1 * trr_02y * trr_10z;
                gout[12] += trr_10x * trr_01y * trr_01z;
                gout[13] += 1 * trr_11y * trr_01z;
                gout[14] += 1 * trr_01y * trr_11z;
                double trr_02z = cpz * trr_01z + 1*b01 * wt;
                gout[15] += trr_10x * fac1 * trr_02z;
                gout[16] += 1 * trr_10y * trr_02z;
                double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                gout[17] += 1 * fac1 * trr_12z;
            }
        }
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        if (reorder_aux) {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 18; ++n) {
                int k = n / 3;
                int ij = n - 3 * k;
                j3c_tensor[ij*naux + k*nksh] = gout[n];
            }
        } else {
            int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 6;
            double *j3c_tensor = out + pair_offset * naux + k0;
            for (int n = 0; n < 18; ++n) {
                int k = n / 3;
                int ij = n - 3 * k;
                j3c_tensor[ij*naux + k] = gout[n];
            }
        }
    }
}

__device__ inline
void int3c2e_112(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int thread_id = threadIdx.x;
    int st_id = thread_id % 64;
    int gout_id = thread_id / 64;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 128;
    double *gy = gx + 768;
    double *gz = gx + 1536;
    double *Rpq = gx + 2304;
    double *rjri = gx + 2496;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 64) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[64] = yjyi;
            rjri[128] = zjzi;
            rjri[192] = rr_ij;
        }
        double gout[14];
        for (int n = 0; n < 14; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        double s0, s1, s2;
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
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[64] * aj_aij + ri[1];
            double zij = rjri[128] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[192];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, 64, gout_id, 4);
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*128];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                double b01 = .5/ak * (1 - rt_ak);
                for (int n = gout_id; n < 3; n += 4) {
                    if (n == 2) {
                        gz[0] = rw[irys*128+64];
                    }
                    double *_gx = gx + n * 768;
                    double xjxi = rjri[n * 64];
                    double Rpa = xjxi * aj_aij;
                    double c0x = Rpa - rt_aij * Rpq[n * 64];
                    s0 = _gx[0];
                    s1 = c0x * s0;
                    _gx[64] = s1;
                    s2 = c0x * s1 + 1 * b10 * s0;
                    _gx[128] = s2;
                    double cpx = rt_ak * Rpq[n * 64];
                    s0 = _gx[0];
                    s1 = cpx * s0;
                    _gx[256] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    _gx[512] = s2;
                    s0 = _gx[64];
                    s1 = cpx * s0;
                    s1 += 1 * b00 * _gx[0];
                    _gx[320] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    s2 += 1 * b00 * _gx[256];
                    _gx[576] = s2;
                    s0 = _gx[128];
                    s1 = cpx * s0;
                    s1 += 2 * b00 * _gx[64];
                    _gx[384] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    s2 += 2 * b00 * _gx[320];
                    _gx[640] = s2;
                    s1 = _gx[128];
                    s0 = _gx[64];
                    _gx[192] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[0];
                    _gx[128] = s1 - xjxi * s0;
                    s1 = _gx[384];
                    s0 = _gx[320];
                    _gx[448] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[256];
                    _gx[384] = s1 - xjxi * s0;
                    s1 = _gx[640];
                    s0 = _gx[576];
                    _gx[704] = s1 - xjxi * s0;
                    s1 = s0;
                    s0 = _gx[512];
                    _gx[640] = s1 - xjxi * s0;
                }
                __syncthreads();
                switch (gout_id) {
                case 0:
                gout[0] += gx[704] * gy[0] * gz[0];
                gout[1] += gx[192] * gy[256] * gz[256];
                gout[2] += gx[384] * gy[64] * gz[256];
                gout[3] += gx[640] * gy[0] * gz[64];
                gout[4] += gx[128] * gy[256] * gz[320];
                gout[5] += gx[320] * gy[128] * gz[256];
                gout[6] += gx[512] * gy[192] * gz[0];
                gout[7] += gx[0] * gy[448] * gz[256];
                gout[8] += gx[256] * gy[128] * gz[320];
                gout[9] += gx[576] * gy[0] * gz[128];
                gout[10] += gx[64] * gy[256] * gz[384];
                gout[11] += gx[256] * gy[64] * gz[384];
                gout[12] += gx[512] * gy[0] * gz[192];
                gout[13] += gx[0] * gy[256] * gz[448];
                break;
                case 1:
                gout[0] += gx[448] * gy[256] * gz[0];
                gout[1] += gx[192] * gy[0] * gz[512];
                gout[2] += gx[128] * gy[576] * gz[0];
                gout[3] += gx[384] * gy[256] * gz[64];
                gout[4] += gx[128] * gy[0] * gz[576];
                gout[5] += gx[64] * gy[640] * gz[0];
                gout[6] += gx[256] * gy[448] * gz[0];
                gout[7] += gx[0] * gy[192] * gz[512];
                gout[8] += gx[0] * gy[640] * gz[64];
                gout[9] += gx[320] * gy[256] * gz[128];
                gout[10] += gx[64] * gy[0] * gz[640];
                gout[11] += gx[0] * gy[576] * gz[128];
                gout[12] += gx[256] * gy[256] * gz[192];
                gout[13] += gx[0] * gy[0] * gz[704];
                break;
                case 2:
                gout[0] += gx[448] * gy[0] * gz[256];
                gout[1] += gx[640] * gy[64] * gz[0];
                gout[2] += gx[128] * gy[320] * gz[256];
                gout[3] += gx[384] * gy[0] * gz[320];
                gout[4] += gx[576] * gy[128] * gz[0];
                gout[5] += gx[64] * gy[384] * gz[256];
                gout[6] += gx[256] * gy[192] * gz[256];
                gout[7] += gx[512] * gy[128] * gz[64];
                gout[8] += gx[0] * gy[384] * gz[320];
                gout[9] += gx[320] * gy[0] * gz[384];
                gout[10] += gx[512] * gy[64] * gz[128];
                gout[11] += gx[0] * gy[320] * gz[384];
                gout[12] += gx[256] * gy[0] * gz[448];
                break;
                case 3:
                gout[0] += gx[192] * gy[512] * gz[0];
                gout[1] += gx[384] * gy[320] * gz[0];
                gout[2] += gx[128] * gy[64] * gz[512];
                gout[3] += gx[128] * gy[512] * gz[64];
                gout[4] += gx[320] * gy[384] * gz[0];
                gout[5] += gx[64] * gy[128] * gz[512];
                gout[6] += gx[0] * gy[704] * gz[0];
                gout[7] += gx[256] * gy[384] * gz[64];
                gout[8] += gx[0] * gy[128] * gz[576];
                gout[9] += gx[64] * gy[512] * gz[128];
                gout[10] += gx[256] * gy[320] * gz[128];
                gout[11] += gx[0] * gy[64] * gz[640];
                gout[12] += gx[0] * gy[512] * gz[192];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            if (reorder_aux) {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 14; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 54) break;
                    int ij = ijk / 6;
                    int k  = ijk - 6 * ij;
                    j3c_tensor[ij*naux + k*nksh] = gout[n];
                }
            } else {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 6;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 14; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 54) break;
                    int ij = ijk / 6;
                    int k  = ijk - 6 * ij;
                    j3c_tensor[ij*naux + k] = gout[n];
                }
            }
        }
    }
}

__device__ inline
void int3c2e_202(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int thread_id = threadIdx.x;
    int st_id = thread_id % 128;
    int gout_id = thread_id / 128;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 256;
    double *gy = gx + 1152;
    double *gz = gx + 2304;
    double *Rpq = gx + 3456;
    double *rjri = gx + 3840;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 128) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[128] = yjyi;
            rjri[256] = zjzi;
            rjri[384] = rr_ij;
        }
        double gout[18];
        for (int n = 0; n < 18; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        double s0, s1, s2;
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
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[128] * aj_aij + ri[1];
            double zij = rjri[256] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[384];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[128] = ypq;
                Rpq[256] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, 128, gout_id, 2);
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*256];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                double b01 = .5/ak * (1 - rt_ak);
                for (int n = gout_id; n < 3; n += 2) {
                    if (n == 2) {
                        gz[0] = rw[irys*256+128];
                    }
                    double *_gx = gx + n * 1152;
                    double xjxi = rjri[n * 128];
                    double Rpa = xjxi * aj_aij;
                    double c0x = Rpa - rt_aij * Rpq[n * 128];
                    s0 = _gx[0];
                    s1 = c0x * s0;
                    _gx[128] = s1;
                    s2 = c0x * s1 + 1 * b10 * s0;
                    _gx[256] = s2;
                    double cpx = rt_ak * Rpq[n * 128];
                    s0 = _gx[0];
                    s1 = cpx * s0;
                    _gx[384] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    _gx[768] = s2;
                    s0 = _gx[128];
                    s1 = cpx * s0;
                    s1 += 1 * b00 * _gx[0];
                    _gx[512] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    s2 += 1 * b00 * _gx[384];
                    _gx[896] = s2;
                    s0 = _gx[256];
                    s1 = cpx * s0;
                    s1 += 2 * b00 * _gx[128];
                    _gx[640] = s1;
                    s2 = cpx*s1 + 1 * b01 *s0;
                    s2 += 2 * b00 * _gx[512];
                    _gx[1024] = s2;
                }
                __syncthreads();
                switch (gout_id) {
                case 0:
                gout[0] += gx[1024] * gy[0] * gz[0];
                gout[1] += gx[640] * gy[0] * gz[384];
                gout[2] += gx[256] * gy[384] * gz[384];
                gout[3] += gx[896] * gy[128] * gz[0];
                gout[4] += gx[512] * gy[128] * gz[384];
                gout[5] += gx[128] * gy[512] * gz[384];
                gout[6] += gx[896] * gy[0] * gz[128];
                gout[7] += gx[512] * gy[0] * gz[512];
                gout[8] += gx[128] * gy[384] * gz[512];
                gout[9] += gx[768] * gy[256] * gz[0];
                gout[10] += gx[384] * gy[256] * gz[384];
                gout[11] += gx[0] * gy[640] * gz[384];
                gout[12] += gx[768] * gy[128] * gz[128];
                gout[13] += gx[384] * gy[128] * gz[512];
                gout[14] += gx[0] * gy[512] * gz[512];
                gout[15] += gx[768] * gy[0] * gz[256];
                gout[16] += gx[384] * gy[0] * gz[640];
                gout[17] += gx[0] * gy[384] * gz[640];
                break;
                case 1:
                gout[0] += gx[640] * gy[384] * gz[0];
                gout[1] += gx[256] * gy[768] * gz[0];
                gout[2] += gx[256] * gy[0] * gz[768];
                gout[3] += gx[512] * gy[512] * gz[0];
                gout[4] += gx[128] * gy[896] * gz[0];
                gout[5] += gx[128] * gy[128] * gz[768];
                gout[6] += gx[512] * gy[384] * gz[128];
                gout[7] += gx[128] * gy[768] * gz[128];
                gout[8] += gx[128] * gy[0] * gz[896];
                gout[9] += gx[384] * gy[640] * gz[0];
                gout[10] += gx[0] * gy[1024] * gz[0];
                gout[11] += gx[0] * gy[256] * gz[768];
                gout[12] += gx[384] * gy[512] * gz[128];
                gout[13] += gx[0] * gy[896] * gz[128];
                gout[14] += gx[0] * gy[128] * gz[896];
                gout[15] += gx[384] * gy[384] * gz[256];
                gout[16] += gx[0] * gy[768] * gz[256];
                gout[17] += gx[0] * gy[0] * gz[1024];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            if (reorder_aux) {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 18; ++n) {
                    int ijk = n*2+gout_id;
                    if (ijk >= 36) break;
                    int ij = ijk / 6;
                    int k  = ijk - 6 * ij;
                    j3c_tensor[ij*naux + k*nksh] = gout[n];
                }
            } else {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 6;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 18; ++n) {
                    int ijk = n*2+gout_id;
                    if (ijk >= 36) break;
                    int ij = ijk / 6;
                    int k  = ijk - 6 * ij;
                    j3c_tensor[ij*naux + k] = gout[n];
                }
            }
        }
    }
}

__device__ inline
void int3c2e_212(double *out, RysIntEnvVars& envs, int shl_pair0, int shl_pair1,
                    int ksh0, int ksh1, int iprim, int jprim, int kprim,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int thread_id = threadIdx.x;
    int st_id = thread_id % 64;
    int gout_id = thread_id / 64;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    int nshl_pair = shl_pair1 - shl_pair0;
    int nksh = ksh1 - ksh0;
    int nst = nshl_pair * nksh;
    int nroots = 3;
    if (omega < 0) {
        nroots *= 2;
    }
    __syncthreads();
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + st_id;
    double *gx = rw + nroots * 128;
    double *gy = gx + 1152;
    double *gz = gx + 2304;
    double *Rpq = gx + 3456;
    double *rjri = gx + 3648;
    if (gout_id == 0) {
        gx[0] = 1.;
    }
    for (int ijk_idx = st_id; ijk_idx < nst+st_id; ijk_idx += 64) {
        int shl_pair_in_block = ijk_idx / nksh;
        int ksh_in_block = ijk_idx - nksh * shl_pair_in_block;
        __syncthreads();
        if (ijk_idx >= nst) {
            shl_pair_in_block = 0;
            if (gout_id == 0) {
                gx[0] = 0.;
            }
        }
        int pair_ij = shl_pair_in_block + shl_pair0;
        int ksh = ksh_in_block + ksh0;
        int bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            rjri[0] = xjxi;
            rjri[64] = yjyi;
            rjri[128] = zjzi;
            rjri[192] = rr_ij;
        }
        double gout[27];
        for (int n = 0; n < 27; ++n) { gout[n] = 0; }
        int ijkprim = iprim * jprim * kprim;
        double s0, s1, s2;
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
            double aij = ai + aj;
            double aj_aij = aj / aij;
            __syncthreads();
            double xij = rjri[0] * aj_aij + ri[0];
            double yij = rjri[64] * aj_aij + ri[1];
            double zij = rjri[128] * aj_aij + ri[2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            if (gout_id == 0) {
                double cijk = ci[ip] * cj[jp] * ck[kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rjri[192];
                gy[0] = fac * exp(-Kab);
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
            }
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;
            double theta = aij * ak / (aij + ak);
            rys_roots_rs(nroots, theta, rr, omega, rw, 64, gout_id, 4);
            for (int irys = 0; irys < nroots; ++irys) {
                __syncthreads();
                double rt = rw[irys*128];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                double b01 = .5/ak * (1 - rt_ak);
                for (int n = gout_id; n < 3; n += 4) {
                    if (n == 2) {
                        gz[0] = rw[irys*128+64];
                    }
                    double *_gx = gx + n * 1152;
                    double xjxi = rjri[n * 64];
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
                switch (gout_id) {
                case 0:
                gout[0] += gx[1088] * gy[0] * gz[0];
                gout[1] += gx[320] * gy[384] * gz[384];
                gout[2] += gx[640] * gy[64] * gz[384];
                gout[3] += gx[1024] * gy[0] * gz[64];
                gout[4] += gx[256] * gy[384] * gz[448];
                gout[5] += gx[576] * gy[128] * gz[384];
                gout[6] += gx[960] * gy[64] * gz[64];
                gout[7] += gx[192] * gy[448] * gz[448];
                gout[8] += gx[576] * gy[0] * gz[512];
                gout[9] += gx[896] * gy[192] * gz[0];
                gout[10] += gx[128] * gy[576] * gz[384];
                gout[11] += gx[448] * gy[256] * gz[384];
                gout[12] += gx[832] * gy[192] * gz[64];
                gout[13] += gx[64] * gy[576] * gz[448];
                gout[14] += gx[384] * gy[320] * gz[384];
                gout[15] += gx[768] * gy[256] * gz[64];
                gout[16] += gx[0] * gy[640] * gz[448];
                gout[17] += gx[384] * gy[192] * gz[512];
                gout[18] += gx[896] * gy[0] * gz[192];
                gout[19] += gx[128] * gy[384] * gz[576];
                gout[20] += gx[448] * gy[64] * gz[576];
                gout[21] += gx[832] * gy[0] * gz[256];
                gout[22] += gx[64] * gy[384] * gz[640];
                gout[23] += gx[384] * gy[128] * gz[576];
                gout[24] += gx[768] * gy[64] * gz[256];
                gout[25] += gx[0] * gy[448] * gz[640];
                gout[26] += gx[384] * gy[0] * gz[704];
                break;
                case 1:
                gout[0] += gx[704] * gy[384] * gz[0];
                gout[1] += gx[320] * gy[0] * gz[768];
                gout[2] += gx[256] * gy[832] * gz[0];
                gout[3] += gx[640] * gy[384] * gz[64];
                gout[4] += gx[256] * gy[0] * gz[832];
                gout[5] += gx[192] * gy[896] * gz[0];
                gout[6] += gx[576] * gy[448] * gz[64];
                gout[7] += gx[192] * gy[64] * gz[832];
                gout[8] += gx[192] * gy[768] * gz[128];
                gout[9] += gx[512] * gy[576] * gz[0];
                gout[10] += gx[128] * gy[192] * gz[768];
                gout[11] += gx[64] * gy[1024] * gz[0];
                gout[12] += gx[448] * gy[576] * gz[64];
                gout[13] += gx[64] * gy[192] * gz[832];
                gout[14] += gx[0] * gy[1088] * gz[0];
                gout[15] += gx[384] * gy[640] * gz[64];
                gout[16] += gx[0] * gy[256] * gz[832];
                gout[17] += gx[0] * gy[960] * gz[128];
                gout[18] += gx[512] * gy[384] * gz[192];
                gout[19] += gx[128] * gy[0] * gz[960];
                gout[20] += gx[64] * gy[832] * gz[192];
                gout[21] += gx[448] * gy[384] * gz[256];
                gout[22] += gx[64] * gy[0] * gz[1024];
                gout[23] += gx[0] * gy[896] * gz[192];
                gout[24] += gx[384] * gy[448] * gz[256];
                gout[25] += gx[0] * gy[64] * gz[1024];
                gout[26] += gx[0] * gy[768] * gz[320];
                break;
                case 2:
                gout[0] += gx[704] * gy[0] * gz[384];
                gout[1] += gx[1024] * gy[64] * gz[0];
                gout[2] += gx[256] * gy[448] * gz[384];
                gout[3] += gx[640] * gy[0] * gz[448];
                gout[4] += gx[960] * gy[128] * gz[0];
                gout[5] += gx[192] * gy[512] * gz[384];
                gout[6] += gx[576] * gy[64] * gz[448];
                gout[7] += gx[960] * gy[0] * gz[128];
                gout[8] += gx[192] * gy[384] * gz[512];
                gout[9] += gx[512] * gy[192] * gz[384];
                gout[10] += gx[832] * gy[256] * gz[0];
                gout[11] += gx[64] * gy[640] * gz[384];
                gout[12] += gx[448] * gy[192] * gz[448];
                gout[13] += gx[768] * gy[320] * gz[0];
                gout[14] += gx[0] * gy[704] * gz[384];
                gout[15] += gx[384] * gy[256] * gz[448];
                gout[16] += gx[768] * gy[192] * gz[128];
                gout[17] += gx[0] * gy[576] * gz[512];
                gout[18] += gx[512] * gy[0] * gz[576];
                gout[19] += gx[832] * gy[64] * gz[192];
                gout[20] += gx[64] * gy[448] * gz[576];
                gout[21] += gx[448] * gy[0] * gz[640];
                gout[22] += gx[768] * gy[128] * gz[192];
                gout[23] += gx[0] * gy[512] * gz[576];
                gout[24] += gx[384] * gy[64] * gz[640];
                gout[25] += gx[768] * gy[0] * gz[320];
                gout[26] += gx[0] * gy[384] * gz[704];
                break;
                case 3:
                gout[0] += gx[320] * gy[768] * gz[0];
                gout[1] += gx[640] * gy[448] * gz[0];
                gout[2] += gx[256] * gy[64] * gz[768];
                gout[3] += gx[256] * gy[768] * gz[64];
                gout[4] += gx[576] * gy[512] * gz[0];
                gout[5] += gx[192] * gy[128] * gz[768];
                gout[6] += gx[192] * gy[832] * gz[64];
                gout[7] += gx[576] * gy[384] * gz[128];
                gout[8] += gx[192] * gy[0] * gz[896];
                gout[9] += gx[128] * gy[960] * gz[0];
                gout[10] += gx[448] * gy[640] * gz[0];
                gout[11] += gx[64] * gy[256] * gz[768];
                gout[12] += gx[64] * gy[960] * gz[64];
                gout[13] += gx[384] * gy[704] * gz[0];
                gout[14] += gx[0] * gy[320] * gz[768];
                gout[15] += gx[0] * gy[1024] * gz[64];
                gout[16] += gx[384] * gy[576] * gz[128];
                gout[17] += gx[0] * gy[192] * gz[896];
                gout[18] += gx[128] * gy[768] * gz[192];
                gout[19] += gx[448] * gy[448] * gz[192];
                gout[20] += gx[64] * gy[64] * gz[960];
                gout[21] += gx[64] * gy[768] * gz[256];
                gout[22] += gx[384] * gy[512] * gz[192];
                gout[23] += gx[0] * gy[128] * gz[960];
                gout[24] += gx[0] * gy[832] * gz[256];
                gout[25] += gx[384] * gy[384] * gz[320];
                gout[26] += gx[0] * gy[0] * gz[1088];
                break;
                }
            }
        }
        if (ijk_idx < nst) {
            size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
            if (reorder_aux) {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 27; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 108) break;
                    int ij = ijk / 6;
                    int k  = ijk - 6 * ij;
                    j3c_tensor[ij*naux + k*nksh] = gout[n];
                }
            } else {
                int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh_in_block * 6;
                double *j3c_tensor = out + pair_offset * naux + k0;
                for (int n = 0; n < 27; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 108) break;
                    int ij = ijk / 6;
                    int k  = ijk - 6 * ij;
                    j3c_tensor[ij*naux + k] = gout[n];
                }
            }
        }
    }
}

__device__ inline
int int3c2e_unrolled(double *out, RysIntEnvVars& envs,
                    int shl_pair0, int shl_pair1, int ksh0, int ksh1,
                    int iprim, int jprim, int kprim, int li, int lj, int lk,
                    double omega, uint32_t *bas_ij_idx, int *ao_pair_loc,
                    int ao_pair_offset, int aux_offset, int naux, int nao,
                    int reorder_aux)
{
    int kij_type = lk*25 + li*5 + lj;
    switch (kij_type) {
    case 0: // li=0 lj=0 lk=0
        int3c2e_000(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 5: // li=1 lj=0 lk=0
        int3c2e_100(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 6: // li=1 lj=1 lk=0
        int3c2e_110(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 10: // li=2 lj=0 lk=0
        int3c2e_200(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 11: // li=2 lj=1 lk=0
        int3c2e_210(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 12: // li=2 lj=2 lk=0
        int3c2e_220(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 25: // li=0 lj=0 lk=1
        int3c2e_001(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 30: // li=1 lj=0 lk=1
        int3c2e_101(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 31: // li=1 lj=1 lk=1
        int3c2e_111(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 35: // li=2 lj=0 lk=1
        int3c2e_201(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 36: // li=2 lj=1 lk=1
        int3c2e_211(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 37: // li=2 lj=2 lk=1
        int3c2e_221(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 50: // li=0 lj=0 lk=2
        int3c2e_002(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 55: // li=1 lj=0 lk=2
        int3c2e_102(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 56: // li=1 lj=1 lk=2
        int3c2e_112(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 60: // li=2 lj=0 lk=2
        int3c2e_202(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    case 61: // li=2 lj=1 lk=2
        int3c2e_212(out, envs, shl_pair0, shl_pair1, ksh0, ksh1, iprim, jprim, kprim,
            omega, bas_ij_idx, ao_pair_loc, ao_pair_offset, aux_offset, naux, nao, reorder_aux); break;
    default: return 0;
    }
    return 1;
}
