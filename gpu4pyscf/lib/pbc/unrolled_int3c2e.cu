#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"


__device__ __forceinline__
void int3c2e_000(double *out, PBCIntEnvVars& envs,
        uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ double rw[];

    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_count = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[1];
#pragma unroll
        for (int n = 0; n < 1; ++n) { gout[n] = 0; }

        for (int img = 0; img < img_count; img++) {
            int img_jk = img_pool[ijk_id+POOL_SIZE*img];
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double aij = ai + aj;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double aj_aij = aj / aij;
                int jL = img_jk / nimgs;
                int kL = img_jk - nimgs * jL;
                double xi = env[ri+0];
                double yi = env[ri+1];
                double zi = env[ri+2];
                double xjxi = env[rj+0] - xi + img_coords[jL*3+0];
                double yjyi = env[rj+1] - yi + img_coords[jL*3+1];
                double zjzi = env[rj+2] - zi + img_coords[jL*3+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double fac1 = fac * exp(-Kab);
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double omega = env[PTR_RANGE_OMEGA];
                double theta_rr = theta * rr;
                rys_roots(1, theta_rr, rw+st_id, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 2*nst_per_block + st_id;
                rys_roots(1, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 1; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 2; ++irys) {
                    double wt = rw[st_id+(2*irys+1)*nst_per_block];
                    gout[0] += 1 * fac1 * wt;
                }
            }
        }

        int bvk_naux = naux * ncells;
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double *j3c = out + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
        for (int k = 0; k < 1; ++k) {
            for (int ij = 0; ij < 1; ++ij) {
                j3c[ij*bvk_naux + k] += gout[k * 1 + ij];
            }
        }
    }
}

__device__ __forceinline__
void int3c2e_100(double *out, PBCIntEnvVars& envs,
        uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ double rw[];

    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_count = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[3];
#pragma unroll
        for (int n = 0; n < 3; ++n) { gout[n] = 0; }

        for (int img = 0; img < img_count; img++) {
            int img_jk = img_pool[ijk_id+POOL_SIZE*img];
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double aij = ai + aj;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double aj_aij = aj / aij;
                int jL = img_jk / nimgs;
                int kL = img_jk - nimgs * jL;
                double xi = env[ri+0];
                double yi = env[ri+1];
                double zi = env[ri+2];
                double xjxi = env[rj+0] - xi + img_coords[jL*3+0];
                double yjyi = env[rj+1] - yi + img_coords[jL*3+1];
                double zjzi = env[rj+2] - zi + img_coords[jL*3+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double fac1 = fac * exp(-Kab);
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double omega = env[PTR_RANGE_OMEGA];
                double theta_rr = theta * rr;
                rys_roots(1, theta_rr, rw+st_id, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 2*nst_per_block + st_id;
                rys_roots(1, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 1; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 2; ++irys) {
                    double wt = rw[st_id+(2*irys+1)*nst_per_block];
                    double rt = rw[st_id+ 2*irys   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    gout[0] += trr_10x * fac1 * wt;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac1;
                    gout[1] += 1 * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout[2] += 1 * fac1 * trr_10z;
                }
            }
        }

        int bvk_naux = naux * ncells;
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double *j3c = out + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
        for (int k = 0; k < 1; ++k) {
            for (int ij = 0; ij < 3; ++ij) {
                j3c[ij*bvk_naux + k] += gout[k * 3 + ij];
            }
        }
    }
}

__device__ __forceinline__
void int3c2e_110(double *out, PBCIntEnvVars& envs,
        uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ double rw[];

    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_count = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[9];
#pragma unroll
        for (int n = 0; n < 9; ++n) { gout[n] = 0; }

        for (int img = 0; img < img_count; img++) {
            int img_jk = img_pool[ijk_id+POOL_SIZE*img];
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double aij = ai + aj;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double aj_aij = aj / aij;
                int jL = img_jk / nimgs;
                int kL = img_jk - nimgs * jL;
                double xi = env[ri+0];
                double yi = env[ri+1];
                double zi = env[ri+2];
                double xjxi = env[rj+0] - xi + img_coords[jL*3+0];
                double yjyi = env[rj+1] - yi + img_coords[jL*3+1];
                double zjzi = env[rj+2] - zi + img_coords[jL*3+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double fac1 = fac * exp(-Kab);
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double omega = env[PTR_RANGE_OMEGA];
                double theta_rr = theta * rr;
                rys_roots(2, theta_rr, rw+st_id, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 4*nst_per_block + st_id;
                rys_roots(2, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[st_id+(2*irys+1)*nst_per_block];
                    double rt = rw[st_id+ 2*irys   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    gout[0] += hrr_110x * fac1 * wt;
                    double hrr_010x = trr_10x - xjxi * 1;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac1;
                    gout[1] += hrr_010x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
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
        }

        int bvk_naux = naux * ncells;
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double *j3c = out + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
        for (int k = 0; k < 1; ++k) {
            for (int ij = 0; ij < 9; ++ij) {
                j3c[ij*bvk_naux + k] += gout[k * 9 + ij];
            }
        }
    }
}

__device__ __forceinline__
void int3c2e_001(double *out, PBCIntEnvVars& envs,
        uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ double rw[];

    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_count = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[3];
#pragma unroll
        for (int n = 0; n < 3; ++n) { gout[n] = 0; }

        for (int img = 0; img < img_count; img++) {
            int img_jk = img_pool[ijk_id+POOL_SIZE*img];
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double aij = ai + aj;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double aj_aij = aj / aij;
                int jL = img_jk / nimgs;
                int kL = img_jk - nimgs * jL;
                double xi = env[ri+0];
                double yi = env[ri+1];
                double zi = env[ri+2];
                double xjxi = env[rj+0] - xi + img_coords[jL*3+0];
                double yjyi = env[rj+1] - yi + img_coords[jL*3+1];
                double zjzi = env[rj+2] - zi + img_coords[jL*3+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double fac1 = fac * exp(-Kab);
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double omega = env[PTR_RANGE_OMEGA];
                double theta_rr = theta * rr;
                rys_roots(1, theta_rr, rw+st_id, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 2*nst_per_block + st_id;
                rys_roots(1, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 1; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 2; ++irys) {
                    double wt = rw[st_id+(2*irys+1)*nst_per_block];
                    double rt = rw[st_id+ 2*irys   *nst_per_block];
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
        }

        int bvk_naux = naux * ncells;
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double *j3c = out + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
        for (int k = 0; k < 3; ++k) {
            for (int ij = 0; ij < 1; ++ij) {
                j3c[ij*bvk_naux + k] += gout[k * 1 + ij];
            }
        }
    }
}

__device__ __forceinline__
void int3c2e_101(double *out, PBCIntEnvVars& envs,
        uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ double rw[];

    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_count = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[9];
#pragma unroll
        for (int n = 0; n < 9; ++n) { gout[n] = 0; }

        for (int img = 0; img < img_count; img++) {
            int img_jk = img_pool[ijk_id+POOL_SIZE*img];
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double aij = ai + aj;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double aj_aij = aj / aij;
                int jL = img_jk / nimgs;
                int kL = img_jk - nimgs * jL;
                double xi = env[ri+0];
                double yi = env[ri+1];
                double zi = env[ri+2];
                double xjxi = env[rj+0] - xi + img_coords[jL*3+0];
                double yjyi = env[rj+1] - yi + img_coords[jL*3+1];
                double zjzi = env[rj+2] - zi + img_coords[jL*3+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double fac1 = fac * exp(-Kab);
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double omega = env[PTR_RANGE_OMEGA];
                double theta_rr = theta * rr;
                rys_roots(2, theta_rr, rw+st_id, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 4*nst_per_block + st_id;
                rys_roots(2, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[st_id+(2*irys+1)*nst_per_block];
                    double rt = rw[st_id+ 2*irys   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    double b00 = .5 * rt_aa;
                    double rt_ak = rt_aa * aij;
                    double cpx = xpq*rt_ak;
                    double rt_aij = rt_aa * ak;
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    gout[0] += trr_11x * fac1 * wt;
                    double trr_01x = cpx * 1;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac1;
                    gout[1] += trr_01x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
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
        }

        int bvk_naux = naux * ncells;
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double *j3c = out + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
        for (int k = 0; k < 3; ++k) {
            for (int ij = 0; ij < 3; ++ij) {
                j3c[ij*bvk_naux + k] += gout[k * 3 + ij];
            }
        }
    }
}

__device__ __forceinline__
void int3c2e_111(double *out, PBCIntEnvVars& envs,
        uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ double rw[];

    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_count = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[27];
#pragma unroll
        for (int n = 0; n < 27; ++n) { gout[n] = 0; }

        for (int img = 0; img < img_count; img++) {
            int img_jk = img_pool[ijk_id+POOL_SIZE*img];
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double aij = ai + aj;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac = PI_FAC * cijk / (aij*ak*sqrt(aij+ak));
                double aj_aij = aj / aij;
                int jL = img_jk / nimgs;
                int kL = img_jk - nimgs * jL;
                double xi = env[ri+0];
                double yi = env[ri+1];
                double zi = env[ri+2];
                double xjxi = env[rj+0] - xi + img_coords[jL*3+0];
                double yjyi = env[rj+1] - yi + img_coords[jL*3+1];
                double zjzi = env[rj+2] - zi + img_coords[jL*3+2];
                double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double fac1 = fac * exp(-Kab);
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double omega = env[PTR_RANGE_OMEGA];
                double theta_rr = theta * rr;
                rys_roots(2, theta_rr, rw+st_id, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 4*nst_per_block + st_id;
                rys_roots(2, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[st_id+(2*irys+1)*nst_per_block];
                    double rt = rw[st_id+ 2*irys   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    double b00 = .5 * rt_aa;
                    double rt_ak = rt_aa * aij;
                    double cpx = xpq*rt_ak;
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double hrr_111x = trr_21x - xjxi * trr_11x;
                    gout[0] += hrr_111x * fac1 * wt;
                    double trr_01x = cpx * 1;
                    double hrr_011x = trr_11x - xjxi * trr_01x;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac1;
                    gout[1] += hrr_011x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
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
        }

        int bvk_naux = naux * ncells;
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double *j3c = out + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
        for (int k = 0; k < 3; ++k) {
            for (int ij = 0; ij < 9; ++ij) {
                j3c[ij*bvk_naux + k] += gout[k * 9 + ij];
            }
        }
    }
}

__device__ __forceinline__
void int3c2e_212(double *out, PBCIntEnvVars& envs,
        uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int thread_id = threadIdx.x;
    constexpr int nst_per_block = 64;
    int st_id = thread_id % nst_per_block;
    int gout_id = thread_id / nst_per_block;
    int ncells = envs.bvk_ncells;
    int bvk_nbas = envs.nbas * ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + 192 + st_id;
    double *gx = shared_memory + 448 + st_id;
    double *rw = shared_memory + 3904 + st_id;
    for (int task_id = st_id; task_id < num_ijk_tasks + st_id; task_id += nst_per_block) {
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info;
        int ijk_id = 0;
        int img_count = 0;
        if (task_id < num_ijk_tasks) {
            ijk_id = rem_task_idx[task_id];
            ijk_task += ijk_id;
            img_count = ijk_task->img_count;
        }
        __shared__ int max_img_count;
        block_max(img_count, max_img_count);
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout[27];
        for (int n = 0; n < 27; ++n) { gout[n] = 0; }
        __syncthreads();
        gx[0] = PI_FAC;
        for (int img = 0; img < max_img_count; img++) {
            int img_jk = 0;
            if (img < img_count) {
                img_jk = img_pool[ijk_id+POOL_SIZE*img];
            }
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                __syncthreads();
                if (gout_id == 0) {
                    int jL = img_jk / nimgs;
                    int kL = img_jk - nimgs * jL;
                    double xi = env[ri+0];
                    double yi = env[ri+1];
                    double zi = env[ri+2];
                    double xjxi = env[rj+0] - xi + img_coords[jL*3+0];
                    double yjyi = env[rj+1] - yi + img_coords[jL*3+1];
                    double zjzi = env[rj+2] - zi + img_coords[jL*3+2];
                    double fac_ij = 0;
                    if (img < img_count && task_id < num_ijk_tasks) {
                        double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                        double fac = cijk / (aij*ak*sqrt(aij+ak));
                        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                        double theta_ij = ai * aj_aij;
                        double Kab = theta_ij * rr_ij;
                        fac_ij = fac * exp(-Kab);
                    }
                    gx[1152] = fac_ij;
                    rjri[0*nst_per_block] = xjxi;
                    rjri[1*nst_per_block] = yjyi;
                    rjri[2*nst_per_block] = zjzi;
                    double xij = xjxi * aj_aij + xi;
                    double yij = yjyi * aj_aij + yi;
                    double zij = zjzi * aj_aij + zi;
                    double xpq = xij - env[rk+0] - img_coords[kL*3+0];
                    double ypq = yij - env[rk+1] - img_coords[kL*3+1];
                    double zpq = zij - env[rk+2] - img_coords[kL*3+2];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                    Rpq[3*nst_per_block] = rr;
                }
                __syncthreads();
                double theta = aij * ak / (aij + ak);
                double omega = env[PTR_RANGE_OMEGA];
                rys_roots_rs(6, theta, Rpq[3*nst_per_block], omega,
                             rw, nst_per_block, gout_id, 4);
                double s0, s1, s2;
                for (int irys = 0; irys < 6; ++irys) {
                    __syncthreads();
                    if (img < img_count && task_id < num_ijk_tasks) {
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
                    }
                    __syncthreads();
                    if (img < img_count && task_id < num_ijk_tasks) {
                        switch (gout_id) {
                        case 0:
                        gout[0] += gx[1088] * gx[1152] * gx[2304];
                        gout[1] += gx[320] * gx[1536] * gx[2688];
                        gout[2] += gx[640] * gx[1216] * gx[2688];
                        gout[3] += gx[1024] * gx[1152] * gx[2368];
                        gout[4] += gx[256] * gx[1536] * gx[2752];
                        gout[5] += gx[576] * gx[1280] * gx[2688];
                        gout[6] += gx[960] * gx[1216] * gx[2368];
                        gout[7] += gx[192] * gx[1600] * gx[2752];
                        gout[8] += gx[576] * gx[1152] * gx[2816];
                        gout[9] += gx[896] * gx[1344] * gx[2304];
                        gout[10] += gx[128] * gx[1728] * gx[2688];
                        gout[11] += gx[448] * gx[1408] * gx[2688];
                        gout[12] += gx[832] * gx[1344] * gx[2368];
                        gout[13] += gx[64] * gx[1728] * gx[2752];
                        gout[14] += gx[384] * gx[1472] * gx[2688];
                        gout[15] += gx[768] * gx[1408] * gx[2368];
                        gout[16] += gx[0] * gx[1792] * gx[2752];
                        gout[17] += gx[384] * gx[1344] * gx[2816];
                        gout[18] += gx[896] * gx[1152] * gx[2496];
                        gout[19] += gx[128] * gx[1536] * gx[2880];
                        gout[20] += gx[448] * gx[1216] * gx[2880];
                        gout[21] += gx[832] * gx[1152] * gx[2560];
                        gout[22] += gx[64] * gx[1536] * gx[2944];
                        gout[23] += gx[384] * gx[1280] * gx[2880];
                        gout[24] += gx[768] * gx[1216] * gx[2560];
                        gout[25] += gx[0] * gx[1600] * gx[2944];
                        gout[26] += gx[384] * gx[1152] * gx[3008];
                        break;
                        case 1:
                        gout[0] += gx[704] * gx[1536] * gx[2304];
                        gout[1] += gx[320] * gx[1152] * gx[3072];
                        gout[2] += gx[256] * gx[1984] * gx[2304];
                        gout[3] += gx[640] * gx[1536] * gx[2368];
                        gout[4] += gx[256] * gx[1152] * gx[3136];
                        gout[5] += gx[192] * gx[2048] * gx[2304];
                        gout[6] += gx[576] * gx[1600] * gx[2368];
                        gout[7] += gx[192] * gx[1216] * gx[3136];
                        gout[8] += gx[192] * gx[1920] * gx[2432];
                        gout[9] += gx[512] * gx[1728] * gx[2304];
                        gout[10] += gx[128] * gx[1344] * gx[3072];
                        gout[11] += gx[64] * gx[2176] * gx[2304];
                        gout[12] += gx[448] * gx[1728] * gx[2368];
                        gout[13] += gx[64] * gx[1344] * gx[3136];
                        gout[14] += gx[0] * gx[2240] * gx[2304];
                        gout[15] += gx[384] * gx[1792] * gx[2368];
                        gout[16] += gx[0] * gx[1408] * gx[3136];
                        gout[17] += gx[0] * gx[2112] * gx[2432];
                        gout[18] += gx[512] * gx[1536] * gx[2496];
                        gout[19] += gx[128] * gx[1152] * gx[3264];
                        gout[20] += gx[64] * gx[1984] * gx[2496];
                        gout[21] += gx[448] * gx[1536] * gx[2560];
                        gout[22] += gx[64] * gx[1152] * gx[3328];
                        gout[23] += gx[0] * gx[2048] * gx[2496];
                        gout[24] += gx[384] * gx[1600] * gx[2560];
                        gout[25] += gx[0] * gx[1216] * gx[3328];
                        gout[26] += gx[0] * gx[1920] * gx[2624];
                        break;
                        case 2:
                        gout[0] += gx[704] * gx[1152] * gx[2688];
                        gout[1] += gx[1024] * gx[1216] * gx[2304];
                        gout[2] += gx[256] * gx[1600] * gx[2688];
                        gout[3] += gx[640] * gx[1152] * gx[2752];
                        gout[4] += gx[960] * gx[1280] * gx[2304];
                        gout[5] += gx[192] * gx[1664] * gx[2688];
                        gout[6] += gx[576] * gx[1216] * gx[2752];
                        gout[7] += gx[960] * gx[1152] * gx[2432];
                        gout[8] += gx[192] * gx[1536] * gx[2816];
                        gout[9] += gx[512] * gx[1344] * gx[2688];
                        gout[10] += gx[832] * gx[1408] * gx[2304];
                        gout[11] += gx[64] * gx[1792] * gx[2688];
                        gout[12] += gx[448] * gx[1344] * gx[2752];
                        gout[13] += gx[768] * gx[1472] * gx[2304];
                        gout[14] += gx[0] * gx[1856] * gx[2688];
                        gout[15] += gx[384] * gx[1408] * gx[2752];
                        gout[16] += gx[768] * gx[1344] * gx[2432];
                        gout[17] += gx[0] * gx[1728] * gx[2816];
                        gout[18] += gx[512] * gx[1152] * gx[2880];
                        gout[19] += gx[832] * gx[1216] * gx[2496];
                        gout[20] += gx[64] * gx[1600] * gx[2880];
                        gout[21] += gx[448] * gx[1152] * gx[2944];
                        gout[22] += gx[768] * gx[1280] * gx[2496];
                        gout[23] += gx[0] * gx[1664] * gx[2880];
                        gout[24] += gx[384] * gx[1216] * gx[2944];
                        gout[25] += gx[768] * gx[1152] * gx[2624];
                        gout[26] += gx[0] * gx[1536] * gx[3008];
                        break;
                        case 3:
                        gout[0] += gx[320] * gx[1920] * gx[2304];
                        gout[1] += gx[640] * gx[1600] * gx[2304];
                        gout[2] += gx[256] * gx[1216] * gx[3072];
                        gout[3] += gx[256] * gx[1920] * gx[2368];
                        gout[4] += gx[576] * gx[1664] * gx[2304];
                        gout[5] += gx[192] * gx[1280] * gx[3072];
                        gout[6] += gx[192] * gx[1984] * gx[2368];
                        gout[7] += gx[576] * gx[1536] * gx[2432];
                        gout[8] += gx[192] * gx[1152] * gx[3200];
                        gout[9] += gx[128] * gx[2112] * gx[2304];
                        gout[10] += gx[448] * gx[1792] * gx[2304];
                        gout[11] += gx[64] * gx[1408] * gx[3072];
                        gout[12] += gx[64] * gx[2112] * gx[2368];
                        gout[13] += gx[384] * gx[1856] * gx[2304];
                        gout[14] += gx[0] * gx[1472] * gx[3072];
                        gout[15] += gx[0] * gx[2176] * gx[2368];
                        gout[16] += gx[384] * gx[1728] * gx[2432];
                        gout[17] += gx[0] * gx[1344] * gx[3200];
                        gout[18] += gx[128] * gx[1920] * gx[2496];
                        gout[19] += gx[448] * gx[1600] * gx[2496];
                        gout[20] += gx[64] * gx[1216] * gx[3264];
                        gout[21] += gx[64] * gx[1920] * gx[2560];
                        gout[22] += gx[384] * gx[1664] * gx[2496];
                        gout[23] += gx[0] * gx[1280] * gx[3264];
                        gout[24] += gx[0] * gx[1984] * gx[2560];
                        gout[25] += gx[384] * gx[1536] * gx[2624];
                        gout[26] += gx[0] * gx[1152] * gx[3392];
                        break;
                        }
                    }
                }
            }
        }
        int bvk_naux = naux * ncells;
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        size_t pair_offset = ao_pair_loc[pair_ij] - ao_pair_offset;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double *j3c = out + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
        if (!to_sph) {
            if (task_id < num_ijk_tasks) {
#pragma unroll
                for (int n = 0; n < 27; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk >= 108) break;
                    int ij = ijk / 6;
                    int k  = ijk - 6 * ij;
                    j3c[ij*bvk_naux + k] += gout[n];
                }
            }
        } else {
            double *inp_local;
            if (55296 < shm_size) {
                inp_local = shared_memory + st_id;
            } else {
                inp_local = c2s_pool + st_id;
            }
            if (task_id < num_ijk_tasks) {
#pragma unroll
                for (int n = 0; n < 27; ++n) {
                    int ijk = n*4+gout_id;
                    if (ijk < 108) {
                        inp_local[ijk*nst_per_block] = gout[n];
                    }
                }
            }
            __syncthreads();
            if (task_id < num_ijk_tasks) {
                double s;
                for (int k = gout_id; k < 6; k += 4) {
                    s = inp_local[nst_per_block*(k+0)];
                    j3c[2*bvk_naux+k] += s*-0.315391565252520002;
                    j3c[4*bvk_naux+k] += s*0.546274215296039535;
                    s = inp_local[nst_per_block*(k+36)];
                    j3c[7*bvk_naux+k] += s*-0.315391565252520002;
                    j3c[9*bvk_naux+k] += s*0.546274215296039535;
                    s = inp_local[nst_per_block*(k+72)];
                    j3c[12*bvk_naux+k] += s*-0.315391565252520002;
                    j3c[14*bvk_naux+k] += s*0.546274215296039535;
                    s = inp_local[nst_per_block*(k+6)];
                    j3c[0*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+42)];
                    j3c[5*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+78)];
                    j3c[10*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+12)];
                    j3c[3*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+48)];
                    j3c[8*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+84)];
                    j3c[13*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+18)];
                    j3c[2*bvk_naux+k] += s*-0.315391565252520002;
                    j3c[4*bvk_naux+k] += s*-0.546274215296039535;
                    s = inp_local[nst_per_block*(k+54)];
                    j3c[7*bvk_naux+k] += s*-0.315391565252520002;
                    j3c[9*bvk_naux+k] += s*-0.546274215296039535;
                    s = inp_local[nst_per_block*(k+90)];
                    j3c[12*bvk_naux+k] += s*-0.315391565252520002;
                    j3c[14*bvk_naux+k] += s*-0.546274215296039535;
                    s = inp_local[nst_per_block*(k+24)];
                    j3c[1*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+60)];
                    j3c[6*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+96)];
                    j3c[11*bvk_naux+k] += s*1.092548430592079070;
                    s = inp_local[nst_per_block*(k+30)];
                    j3c[2*bvk_naux+k] += s*0.630783130505040012;
                    s = inp_local[nst_per_block*(k+66)];
                    j3c[7*bvk_naux+k] += s*0.630783130505040012;
                    s = inp_local[nst_per_block*(k+102)];
                    j3c[12*bvk_naux+k] += s*0.630783130505040012;
                }
            }
            __syncthreads();
        }
    }
}

__device__ inline
int int3c2e_unrolled(double *out, PBCIntEnvVars& envs,
                     uint32_t *img_pool, uint32_t *rem_task_idx, int num_ijk_tasks,
                     ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
                     int shm_size, int iprim, int jprim, int kprim, int li, int lj, int lk,
                     uint32_t *bas_ij_idx, int *ao_pair_loc,
                     int ao_pair_offset, int aux_offset, int nauxbas, int naux, int to_sph)
{
    int kij_type = lk*25 + li*5 + lj;
    switch (kij_type) {
    case 0: // li=0 lj=0 lk=0
        int3c2e_000(out, envs, img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info,
            c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 5: // li=1 lj=0 lk=0
        int3c2e_100(out, envs, img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info,
            c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 6: // li=1 lj=1 lk=0
        int3c2e_110(out, envs, img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info,
            c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 30: // li=1 lj=0 lk=1
        int3c2e_101(out, envs, img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info,
            c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 31: // li=1 lj=1 lk=1
        int3c2e_111(out, envs, img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info,
            c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 61: // li=2 lj=1 lk=2
        int3c2e_212(out, envs, img_pool, rem_task_idx, num_ijk_tasks, ijk_tasks_info,
            c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    default: return 0;
    }
    return 1;
}
