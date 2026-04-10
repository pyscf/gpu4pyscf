#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"


__device__ __forceinline__
void int3c2e_000(double *out, PBCIntEnvVars& envs, uint32_t *img_pool,
        uint32_t *rem_task_idx, int num_ijk_tasks, int img_tile_size,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ int _img_count[];
    double *rw = (double *)(_img_count + 256);
    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        _img_count[st_id] = ijk_task->img_count;
        int img_count = _img_count[st_id];
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
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
            for (int img = 0; img < img_tile_size; img++) {
                int img_jk = 0;
                if (img < img_count) {
                    img_jk = img_pool[ijk_id+POOL_SIZE*(img_count-1-img)];
                    fac = 0;
                }
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
                double omega = env[PTR_RANGE_OMEGA];
                double theta = aij * ak / (aij + ak);
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
        ijk_task->img_count = img_count - img_tile_size;

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
void int3c2e_100(double *out, PBCIntEnvVars& envs, uint32_t *img_pool,
        uint32_t *rem_task_idx, int num_ijk_tasks, int img_tile_size,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ int _img_count[];
    double *rw = (double *)(_img_count + 256);
    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        _img_count[st_id] = ijk_task->img_count;
        int img_count = _img_count[st_id];
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
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
            for (int img = 0; img < img_tile_size; img++) {
                int img_jk = 0;
                if (img < img_count) {
                    img_jk = img_pool[ijk_id+POOL_SIZE*(img_count-1-img)];
                    fac = 0;
                }
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
                double omega = env[PTR_RANGE_OMEGA];
                double theta = aij * ak / (aij + ak);
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
        ijk_task->img_count = img_count - img_tile_size;

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
void int3c2e_110(double *out, PBCIntEnvVars& envs, uint32_t *img_pool,
        uint32_t *rem_task_idx, int num_ijk_tasks, int img_tile_size,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ int _img_count[];
    double *rw = (double *)(_img_count + 256);
    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        _img_count[st_id] = ijk_task->img_count;
        int img_count = _img_count[st_id];
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
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
            for (int img = 0; img < img_tile_size; img++) {
                int img_jk = 0;
                if (img < img_count) {
                    img_jk = img_pool[ijk_id+POOL_SIZE*(img_count-1-img)];
                    fac = 0;
                }
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
                double omega = env[PTR_RANGE_OMEGA];
                double theta = aij * ak / (aij + ak);
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
        ijk_task->img_count = img_count - img_tile_size;

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
void int3c2e_001(double *out, PBCIntEnvVars& envs, uint32_t *img_pool,
        uint32_t *rem_task_idx, int num_ijk_tasks, int img_tile_size,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ int _img_count[];
    double *rw = (double *)(_img_count + 256);
    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        _img_count[st_id] = ijk_task->img_count;
        int img_count = _img_count[st_id];
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
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
            for (int img = 0; img < img_tile_size; img++) {
                int img_jk = 0;
                if (img < img_count) {
                    img_jk = img_pool[ijk_id+POOL_SIZE*(img_count-1-img)];
                    fac = 0;
                }
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
                double omega = env[PTR_RANGE_OMEGA];
                double theta = aij * ak / (aij + ak);
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
        ijk_task->img_count = img_count - img_tile_size;

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
void int3c2e_101(double *out, PBCIntEnvVars& envs, uint32_t *img_pool,
        uint32_t *rem_task_idx, int num_ijk_tasks, int img_tile_size,
        ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
        int shm_size, int iprim, int jprim, int kprim, uint32_t *bas_ij_idx,
        int *ao_pair_loc, int ao_pair_offset, int aux_offset,
        int nauxbas, int naux, int to_sph)
{
    int st_id = threadIdx.x;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    extern __shared__ int _img_count[];
    double *rw = (double *)(_img_count + 256);
    for (int task_id = st_id; task_id < num_ijk_tasks; task_id += nst_per_block) {
        int ijk_id = rem_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        _img_count[st_id] = ijk_task->img_count;
        int img_count = _img_count[st_id];
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
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
            for (int img = 0; img < img_tile_size; img++) {
                int img_jk = 0;
                if (img < img_count) {
                    img_jk = img_pool[ijk_id+POOL_SIZE*(img_count-1-img)];
                    fac = 0;
                }
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
                double omega = env[PTR_RANGE_OMEGA];
                double theta = aij * ak / (aij + ak);
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
        ijk_task->img_count = img_count - img_tile_size;

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

__device__ inline
int int3c2e_unrolled(double *out, PBCIntEnvVars& envs, uint32_t *img_pool,
                     uint32_t *rem_task_idx, int num_ijk_tasks, int img_tile_size,
                     ShellTripletTaskInfo *ijk_tasks_info, double *c2s_pool,
                     int shm_size, int iprim, int jprim, int kprim, int li, int lj, int lk,
                     uint32_t *bas_ij_idx, int *ao_pair_loc,
                     int ao_pair_offset, int aux_offset, int nauxbas, int naux, int to_sph)
{
    int kij_type = lk*25 + li*5 + lj;
    switch (kij_type) {
    case 0: // li=0 lj=0 lk=0
        int3c2e_000(out, envs, img_pool, rem_task_idx, num_ijk_tasks, img_tile_size,
            ijk_tasks_info, c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 5: // li=1 lj=0 lk=0
        int3c2e_100(out, envs, img_pool, rem_task_idx, num_ijk_tasks, img_tile_size,
            ijk_tasks_info, c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 6: // li=1 lj=1 lk=0
        int3c2e_110(out, envs, img_pool, rem_task_idx, num_ijk_tasks, img_tile_size,
            ijk_tasks_info, c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 25: // li=0 lj=0 lk=1
        int3c2e_001(out, envs, img_pool, rem_task_idx, num_ijk_tasks, img_tile_size,
            ijk_tasks_info, c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    case 30: // li=1 lj=0 lk=1
        int3c2e_101(out, envs, img_pool, rem_task_idx, num_ijk_tasks, img_tile_size,
            ijk_tasks_info, c2s_pool, shm_size, iprim, jprim, kprim, bas_ij_idx, ao_pair_loc,
            ao_pair_offset, aux_offset, nauxbas, naux, to_sph); break;
    default: return 0;
    }
    return 1;
}
