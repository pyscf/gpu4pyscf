#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "int3c2e.cuh"


#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void int3c2e_000(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 16 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[16];

    int ntasks = nksh * 16 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 512) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(1, theta_rr, rw1, 512, 0, 1);
                rys_roots(1, theta_fac*theta_rr, rw, 512, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *512] *= theta_fac;
                    rw[(irys*2+1)*512] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*512] *= fac;
                }
                for (int irys = 0; irys < 2; ++irys) {
                    double wt = rw[(2*irys+1)*512];
                    gout0 += 1 * 1 * wt;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*1*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
        }
    }
}

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void int3c2e_100(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 16 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[16];

    int ntasks = nksh * 16 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 512) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(1, theta_rr, rw1, 512, 0, 1);
                rys_roots(1, theta_fac*theta_rr, rw, 512, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *512] *= theta_fac;
                    rw[(irys*2+1)*512] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*512] *= fac;
                }
                for (int irys = 0; irys < 2; ++irys) {
                    double wt = rw[(2*irys+1)*512];
                    double rt = rw[ 2*irys   *512];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    gout0 += trr_10x * 1 * wt;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += 1 * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += 1 * 1 * trr_10z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*3*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
        }
    }
}

__global__
void int3c2e_110(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(2, theta_rr, rw1, 256, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    gout0 += hrr_110x * 1 * wt;
                    double hrr_010x = trr_10x - xjxi * 1;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += hrr_010x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_010x * 1 * trr_10z;
                    double hrr_010y = trr_10y - yjyi * 1;
                    gout3 += trr_10x * hrr_010y * wt;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double hrr_110y = trr_20y - yjyi * trr_10y;
                    gout4 += 1 * hrr_110y * wt;
                    gout5 += 1 * hrr_010y * trr_10z;
                    double hrr_010z = trr_10z - zjzi * wt;
                    gout6 += trr_10x * 1 * hrr_010z;
                    gout7 += 1 * trr_10y * hrr_010z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double hrr_110z = trr_20z - zjzi * trr_10z;
                    gout8 += 1 * 1 * hrr_110z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*9*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
        }
    }
}

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void int3c2e_200(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 16 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[16];

    int ntasks = nksh * 16 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 512) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 2048;
                rys_roots(2, theta_rr, rw1, 512, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 512, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *512] *= theta_fac;
                    rw[(irys*2+1)*512] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*512] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*512];
                    double rt = rw[ 2*irys   *512];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    gout0 += trr_20x * 1 * wt;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += trr_10x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_10x * 1 * trr_10z;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout3 += 1 * trr_20y * wt;
                    gout4 += 1 * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += 1 * 1 * trr_20z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*6*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
        }
    }
}

__global__
void int3c2e_210(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(2, theta_rr, rw1, 256, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double hrr_210x = trr_30x - xjxi * trr_20x;
                    gout0 += hrr_210x * 1 * wt;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += hrr_110x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_110x * 1 * trr_10z;
                    double hrr_010x = trr_10x - xjxi * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout3 += hrr_010x * trr_20y * wt;
                    gout4 += hrr_010x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_010x * 1 * trr_20z;
                    double hrr_010y = trr_10y - yjyi * 1;
                    gout6 += trr_20x * hrr_010y * wt;
                    double hrr_110y = trr_20y - yjyi * trr_10y;
                    gout7 += trr_10x * hrr_110y * wt;
                    gout8 += trr_10x * hrr_010y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_210y = trr_30y - yjyi * trr_20y;
                    gout9 += 1 * hrr_210y * wt;
                    gout10 += 1 * hrr_110y * trr_10z;
                    gout11 += 1 * hrr_010y * trr_20z;
                    double hrr_010z = trr_10z - zjzi * wt;
                    gout12 += trr_20x * 1 * hrr_010z;
                    gout13 += trr_10x * trr_10y * hrr_010z;
                    double hrr_110z = trr_20z - zjzi * trr_10z;
                    gout14 += trr_10x * 1 * hrr_110z;
                    gout15 += 1 * trr_20y * hrr_010z;
                    gout16 += 1 * trr_10y * hrr_110z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_210z = trr_30z - zjzi * trr_20z;
                    gout17 += 1 * 1 * hrr_210z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*18*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
        }
    }
}

__global__
void int3c2e_220(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1536;
                rys_roots(3, theta_rr, rw1, 256, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 6; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    double hrr_310x = trr_40x - xjxi * trr_30x;
                    double hrr_210x = trr_30x - xjxi * trr_20x;
                    double hrr_220x = hrr_310x - xjxi * hrr_210x;
                    gout0 += hrr_220x * 1 * wt;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    double hrr_120x = hrr_210x - xjxi * hrr_110x;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += hrr_120x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_120x * 1 * trr_10z;
                    double hrr_010x = trr_10x - xjxi * 1;
                    double hrr_020x = hrr_110x - xjxi * hrr_010x;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout3 += hrr_020x * trr_20y * wt;
                    gout4 += hrr_020x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_020x * 1 * trr_20z;
                    double hrr_010y = trr_10y - yjyi * 1;
                    gout6 += hrr_210x * hrr_010y * wt;
                    double hrr_110y = trr_20y - yjyi * trr_10y;
                    gout7 += hrr_110x * hrr_110y * wt;
                    gout8 += hrr_110x * hrr_010y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_210y = trr_30y - yjyi * trr_20y;
                    gout9 += hrr_010x * hrr_210y * wt;
                    gout10 += hrr_010x * hrr_110y * trr_10z;
                    gout11 += hrr_010x * hrr_010y * trr_20z;
                    double hrr_010z = trr_10z - zjzi * wt;
                    gout12 += hrr_210x * 1 * hrr_010z;
                    gout13 += hrr_110x * trr_10y * hrr_010z;
                    double hrr_110z = trr_20z - zjzi * trr_10z;
                    gout14 += hrr_110x * 1 * hrr_110z;
                    gout15 += hrr_010x * trr_20y * hrr_010z;
                    gout16 += hrr_010x * trr_10y * hrr_110z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_210z = trr_30z - zjzi * trr_20z;
                    gout17 += hrr_010x * 1 * hrr_210z;
                    double hrr_020y = hrr_110y - yjyi * hrr_010y;
                    gout18 += trr_20x * hrr_020y * wt;
                    double hrr_120y = hrr_210y - yjyi * hrr_110y;
                    gout19 += trr_10x * hrr_120y * wt;
                    gout20 += trr_10x * hrr_020y * trr_10z;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    double hrr_310y = trr_40y - yjyi * trr_30y;
                    double hrr_220y = hrr_310y - yjyi * hrr_210y;
                    gout21 += 1 * hrr_220y * wt;
                    gout22 += 1 * hrr_120y * trr_10z;
                    gout23 += 1 * hrr_020y * trr_20z;
                    gout24 += trr_20x * hrr_010y * hrr_010z;
                    gout25 += trr_10x * hrr_110y * hrr_010z;
                    gout26 += trr_10x * hrr_010y * hrr_110z;
                    gout27 += 1 * hrr_210y * hrr_010z;
                    gout28 += 1 * hrr_110y * hrr_110z;
                    gout29 += 1 * hrr_010y * hrr_210z;
                    double hrr_020z = hrr_110z - zjzi * hrr_010z;
                    gout30 += trr_20x * 1 * hrr_020z;
                    gout31 += trr_10x * trr_10y * hrr_020z;
                    double hrr_120z = hrr_210z - zjzi * hrr_110z;
                    gout32 += trr_10x * 1 * hrr_120z;
                    gout33 += 1 * trr_20y * hrr_020z;
                    gout34 += 1 * trr_10y * hrr_120z;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    double hrr_310z = trr_40z - zjzi * trr_30z;
                    double hrr_220z = hrr_310z - zjzi * hrr_210z;
                    gout35 += 1 * 1 * hrr_220z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*36*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+18*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+19*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+20*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+21*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+22*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+23*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+24*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+25*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+26*n_ctr_pairs, gout26);
            atomicAdd(eri_tensor+27*n_ctr_pairs, gout27);
            atomicAdd(eri_tensor+28*n_ctr_pairs, gout28);
            atomicAdd(eri_tensor+29*n_ctr_pairs, gout29);
            atomicAdd(eri_tensor+30*n_ctr_pairs, gout30);
            atomicAdd(eri_tensor+31*n_ctr_pairs, gout31);
            atomicAdd(eri_tensor+32*n_ctr_pairs, gout32);
            atomicAdd(eri_tensor+33*n_ctr_pairs, gout33);
            atomicAdd(eri_tensor+34*n_ctr_pairs, gout34);
            atomicAdd(eri_tensor+35*n_ctr_pairs, gout35);
        }
    }
}

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void int3c2e_001(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 16 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[16];

    int ntasks = nksh * 16 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 512) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(1, theta_rr, rw1, 512, 0, 1);
                rys_roots(1, theta_fac*theta_rr, rw, 512, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 1; ++irys) {
                    rw[ irys*2   *512] *= theta_fac;
                    rw[(irys*2+1)*512] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*512] *= fac;
                }
                for (int irys = 0; irys < 2; ++irys) {
                    double wt = rw[(2*irys+1)*512];
                    double rt = rw[ 2*irys   *512];
                    double rt_aa = rt / (aij + ak);
                    double rt_ak = rt_aa * aij;
                    double cpx = xpq*rt_ak;
                    double trr_01x = cpx * 1;
                    gout0 += trr_01x * 1 * wt;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout1 += 1 * trr_01y * wt;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout2 += 1 * 1 * trr_01z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*1*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
        }
    }
}

__global__
void int3c2e_101(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(2, theta_rr, rw1, 256, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
                    double rt_aa = rt / (aij + ak);
                    double b00 = .5 * rt_aa;
                    double rt_ak = rt_aa * aij;
                    double cpx = xpq*rt_ak;
                    double rt_aij = rt_aa * ak;
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    gout0 += trr_11x * 1 * wt;
                    double trr_01x = cpx * 1;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += trr_01x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_01x * 1 * trr_10z;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout3 += trr_10x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout4 += 1 * trr_11y * wt;
                    gout5 += 1 * trr_01y * trr_10z;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout6 += trr_10x * 1 * trr_01z;
                    gout7 += 1 * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout8 += 1 * 1 * trr_11z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*3*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
        }
    }
}

__global__
void int3c2e_111(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(2, theta_rr, rw1, 256, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
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
                    gout0 += hrr_111x * 1 * wt;
                    double trr_01x = cpx * 1;
                    double hrr_011x = trr_11x - xjxi * trr_01x;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += hrr_011x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_011x * 1 * trr_10z;
                    double hrr_010y = trr_10y - yjyi * 1;
                    gout3 += trr_11x * hrr_010y * wt;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double hrr_110y = trr_20y - yjyi * trr_10y;
                    gout4 += trr_01x * hrr_110y * wt;
                    gout5 += trr_01x * hrr_010y * trr_10z;
                    double hrr_010z = trr_10z - zjzi * wt;
                    gout6 += trr_11x * 1 * hrr_010z;
                    gout7 += trr_01x * trr_10y * hrr_010z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double hrr_110z = trr_20z - zjzi * trr_10z;
                    gout8 += trr_01x * 1 * hrr_110z;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout9 += hrr_110x * trr_01y * wt;
                    double hrr_010x = trr_10x - xjxi * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout10 += hrr_010x * trr_11y * wt;
                    gout11 += hrr_010x * trr_01y * trr_10z;
                    double hrr_011y = trr_11y - yjyi * trr_01y;
                    gout12 += trr_10x * hrr_011y * wt;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double hrr_111y = trr_21y - yjyi * trr_11y;
                    gout13 += 1 * hrr_111y * wt;
                    gout14 += 1 * hrr_011y * trr_10z;
                    gout15 += trr_10x * trr_01y * hrr_010z;
                    gout16 += 1 * trr_11y * hrr_010z;
                    gout17 += 1 * trr_01y * hrr_110z;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout18 += hrr_110x * 1 * trr_01z;
                    gout19 += hrr_010x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout20 += hrr_010x * 1 * trr_11z;
                    gout21 += trr_10x * hrr_010y * trr_01z;
                    gout22 += 1 * hrr_110y * trr_01z;
                    gout23 += 1 * hrr_010y * trr_11z;
                    double hrr_011z = trr_11z - zjzi * trr_01z;
                    gout24 += trr_10x * 1 * hrr_011z;
                    gout25 += 1 * trr_10y * hrr_011z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double hrr_111z = trr_21z - zjzi * trr_11z;
                    gout26 += 1 * 1 * hrr_111z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*9*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+18*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+19*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+20*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+21*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+22*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+23*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+24*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+25*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+26*n_ctr_pairs, gout26);
        }
    }
}

__global__
void int3c2e_201(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(2, theta_rr, rw1, 256, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
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
                    gout0 += trr_21x * 1 * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += trr_11x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_11x * 1 * trr_10z;
                    double trr_01x = cpx * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout3 += trr_01x * trr_20y * wt;
                    gout4 += trr_01x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += trr_01x * 1 * trr_20z;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout6 += trr_20x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout7 += trr_10x * trr_11y * wt;
                    gout8 += trr_10x * trr_01y * trr_10z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout9 += 1 * trr_21y * wt;
                    gout10 += 1 * trr_11y * trr_10z;
                    gout11 += 1 * trr_01y * trr_20z;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout12 += trr_20x * 1 * trr_01z;
                    gout13 += trr_10x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout14 += trr_10x * 1 * trr_11z;
                    gout15 += 1 * trr_20y * trr_01z;
                    gout16 += 1 * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout17 += 1 * 1 * trr_21z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*6*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
        }
    }
}

__global__
void int3c2e_211(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        double gout36 = 0;
        double gout37 = 0;
        double gout38 = 0;
        double gout39 = 0;
        double gout40 = 0;
        double gout41 = 0;
        double gout42 = 0;
        double gout43 = 0;
        double gout44 = 0;
        double gout45 = 0;
        double gout46 = 0;
        double gout47 = 0;
        double gout48 = 0;
        double gout49 = 0;
        double gout50 = 0;
        double gout51 = 0;
        double gout52 = 0;
        double gout53 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1536;
                rys_roots(3, theta_rr, rw1, 256, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 6; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
                    double rt_aa = rt / (aij + ak);
                    double b00 = .5 * rt_aa;
                    double rt_ak = rt_aa * aij;
                    double cpx = xpq*rt_ak;
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double hrr_211x = trr_31x - xjxi * trr_21x;
                    gout0 += hrr_211x * 1 * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double hrr_111x = trr_21x - xjxi * trr_11x;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += hrr_111x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_111x * 1 * trr_10z;
                    double trr_01x = cpx * 1;
                    double hrr_011x = trr_11x - xjxi * trr_01x;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout3 += hrr_011x * trr_20y * wt;
                    gout4 += hrr_011x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_011x * 1 * trr_20z;
                    double hrr_010y = trr_10y - yjyi * 1;
                    gout6 += trr_21x * hrr_010y * wt;
                    double hrr_110y = trr_20y - yjyi * trr_10y;
                    gout7 += trr_11x * hrr_110y * wt;
                    gout8 += trr_11x * hrr_010y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_210y = trr_30y - yjyi * trr_20y;
                    gout9 += trr_01x * hrr_210y * wt;
                    gout10 += trr_01x * hrr_110y * trr_10z;
                    gout11 += trr_01x * hrr_010y * trr_20z;
                    double hrr_010z = trr_10z - zjzi * wt;
                    gout12 += trr_21x * 1 * hrr_010z;
                    gout13 += trr_11x * trr_10y * hrr_010z;
                    double hrr_110z = trr_20z - zjzi * trr_10z;
                    gout14 += trr_11x * 1 * hrr_110z;
                    gout15 += trr_01x * trr_20y * hrr_010z;
                    gout16 += trr_01x * trr_10y * hrr_110z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_210z = trr_30z - zjzi * trr_20z;
                    gout17 += trr_01x * 1 * hrr_210z;
                    double hrr_210x = trr_30x - xjxi * trr_20x;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout18 += hrr_210x * trr_01y * wt;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout19 += hrr_110x * trr_11y * wt;
                    gout20 += hrr_110x * trr_01y * trr_10z;
                    double hrr_010x = trr_10x - xjxi * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout21 += hrr_010x * trr_21y * wt;
                    gout22 += hrr_010x * trr_11y * trr_10z;
                    gout23 += hrr_010x * trr_01y * trr_20z;
                    double hrr_011y = trr_11y - yjyi * trr_01y;
                    gout24 += trr_20x * hrr_011y * wt;
                    double hrr_111y = trr_21y - yjyi * trr_11y;
                    gout25 += trr_10x * hrr_111y * wt;
                    gout26 += trr_10x * hrr_011y * trr_10z;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double hrr_211y = trr_31y - yjyi * trr_21y;
                    gout27 += 1 * hrr_211y * wt;
                    gout28 += 1 * hrr_111y * trr_10z;
                    gout29 += 1 * hrr_011y * trr_20z;
                    gout30 += trr_20x * trr_01y * hrr_010z;
                    gout31 += trr_10x * trr_11y * hrr_010z;
                    gout32 += trr_10x * trr_01y * hrr_110z;
                    gout33 += 1 * trr_21y * hrr_010z;
                    gout34 += 1 * trr_11y * hrr_110z;
                    gout35 += 1 * trr_01y * hrr_210z;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout36 += hrr_210x * 1 * trr_01z;
                    gout37 += hrr_110x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout38 += hrr_110x * 1 * trr_11z;
                    gout39 += hrr_010x * trr_20y * trr_01z;
                    gout40 += hrr_010x * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout41 += hrr_010x * 1 * trr_21z;
                    gout42 += trr_20x * hrr_010y * trr_01z;
                    gout43 += trr_10x * hrr_110y * trr_01z;
                    gout44 += trr_10x * hrr_010y * trr_11z;
                    gout45 += 1 * hrr_210y * trr_01z;
                    gout46 += 1 * hrr_110y * trr_11z;
                    gout47 += 1 * hrr_010y * trr_21z;
                    double hrr_011z = trr_11z - zjzi * trr_01z;
                    gout48 += trr_20x * 1 * hrr_011z;
                    gout49 += trr_10x * trr_10y * hrr_011z;
                    double hrr_111z = trr_21z - zjzi * trr_11z;
                    gout50 += trr_10x * 1 * hrr_111z;
                    gout51 += 1 * trr_20y * hrr_011z;
                    gout52 += 1 * trr_10y * hrr_111z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double hrr_211z = trr_31z - zjzi * trr_21z;
                    gout53 += 1 * 1 * hrr_211z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*18*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+18*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+19*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+20*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+21*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+22*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+23*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+24*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+25*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+26*n_ctr_pairs, gout26);
            atomicAdd(eri_tensor+27*n_ctr_pairs, gout27);
            atomicAdd(eri_tensor+28*n_ctr_pairs, gout28);
            atomicAdd(eri_tensor+29*n_ctr_pairs, gout29);
            atomicAdd(eri_tensor+30*n_ctr_pairs, gout30);
            atomicAdd(eri_tensor+31*n_ctr_pairs, gout31);
            atomicAdd(eri_tensor+32*n_ctr_pairs, gout32);
            atomicAdd(eri_tensor+33*n_ctr_pairs, gout33);
            atomicAdd(eri_tensor+34*n_ctr_pairs, gout34);
            atomicAdd(eri_tensor+35*n_ctr_pairs, gout35);
            atomicAdd(eri_tensor+36*n_ctr_pairs, gout36);
            atomicAdd(eri_tensor+37*n_ctr_pairs, gout37);
            atomicAdd(eri_tensor+38*n_ctr_pairs, gout38);
            atomicAdd(eri_tensor+39*n_ctr_pairs, gout39);
            atomicAdd(eri_tensor+40*n_ctr_pairs, gout40);
            atomicAdd(eri_tensor+41*n_ctr_pairs, gout41);
            atomicAdd(eri_tensor+42*n_ctr_pairs, gout42);
            atomicAdd(eri_tensor+43*n_ctr_pairs, gout43);
            atomicAdd(eri_tensor+44*n_ctr_pairs, gout44);
            atomicAdd(eri_tensor+45*n_ctr_pairs, gout45);
            atomicAdd(eri_tensor+46*n_ctr_pairs, gout46);
            atomicAdd(eri_tensor+47*n_ctr_pairs, gout47);
            atomicAdd(eri_tensor+48*n_ctr_pairs, gout48);
            atomicAdd(eri_tensor+49*n_ctr_pairs, gout49);
            atomicAdd(eri_tensor+50*n_ctr_pairs, gout50);
            atomicAdd(eri_tensor+51*n_ctr_pairs, gout51);
            atomicAdd(eri_tensor+52*n_ctr_pairs, gout52);
            atomicAdd(eri_tensor+53*n_ctr_pairs, gout53);
        }
    }
}

__global__
void int3c2e_221(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 2 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    double *gx = rw + 768;
    double *gy = gx + 1152;
    double *gz = gy + 1152;
    double *rjri = gz + 1152;
    double *Rpq = rjri + 192;
    __shared__ int img_counts_in_warp[WARPS];

    int ntasks = nksh * 2 * SPTAKS_PER_BLOCK;
    for (int task0 = 0; task0 < ntasks; task0 += 64) {
        int ijk_idx = task0 + ksp_id;
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();
        gy[0] = 1.;

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double s0, s1, s2;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            __syncthreads();
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                if (gout_id == 0) {
                    gy[0] = 0.;
                }
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            if (gout_id == 0) {
                double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
                double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
                double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rjri[0] = xjxi;
                rjri[64] = yjyi;
                rjri[128] = zjzi;
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
                Rpq[192] = rr;
            }
            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                __syncthreads();
                if (gout_id == 0) {
                    double cijk = fac_ij * ck[kp];
                    gx[0] = cijk / (aij*ak*sqrt(aij+ak));
                }
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * Rpq[192];
                double *rw1 = rw + 384;
                __syncthreads();
                rys_roots(3, theta_rr, rw1, 64, gout_id, 4);
                rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                __syncthreads();
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = gout_id; irys < 3; irys += 4) {
                    rw[ irys*2   *64] *= theta_fac;
                    rw[(irys*2+1)*64] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 6; ++irys) {
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
                    gout0 += gx[1088] * gy[0] * gz[0];
                    gout1 += gx[960] * gy[64] * gz[64];
                    gout2 += gx[832] * gy[192] * gz[64];
                    gout3 += gx[896] * gy[0] * gz[192];
                    gout4 += gx[768] * gy[64] * gz[256];
                    gout5 += gx[640] * gy[384] * gz[64];
                    gout6 += gx[704] * gy[192] * gz[192];
                    gout7 += gx[576] * gy[256] * gz[256];
                    gout8 += gx[640] * gy[0] * gz[448];
                    gout9 += gx[512] * gy[576] * gz[0];
                    gout10 += gx[384] * gy[640] * gz[64];
                    gout11 += gx[256] * gy[768] * gz[64];
                    gout12 += gx[320] * gy[576] * gz[192];
                    gout13 += gx[192] * gy[640] * gz[256];
                    gout14 += gx[64] * gy[960] * gz[64];
                    gout15 += gx[128] * gy[768] * gz[192];
                    gout16 += gx[0] * gy[832] * gz[256];
                    gout17 += gx[64] * gy[576] * gz[448];
                    gout18 += gx[512] * gy[0] * gz[576];
                    gout19 += gx[384] * gy[64] * gz[640];
                    gout20 += gx[256] * gy[192] * gz[640];
                    gout21 += gx[320] * gy[0] * gz[768];
                    gout22 += gx[192] * gy[64] * gz[832];
                    gout23 += gx[64] * gy[384] * gz[640];
                    gout24 += gx[128] * gy[192] * gz[768];
                    gout25 += gx[0] * gy[256] * gz[832];
                    gout26 += gx[64] * gy[0] * gz[1024];
                    break;
                    case 1:
                    gout0 += gx[1024] * gy[64] * gz[0];
                    gout1 += gx[960] * gy[0] * gz[128];
                    gout2 += gx[768] * gy[320] * gz[0];
                    gout3 += gx[832] * gy[64] * gz[192];
                    gout4 += gx[768] * gy[0] * gz[320];
                    gout5 += gx[576] * gy[512] * gz[0];
                    gout6 += gx[640] * gy[256] * gz[192];
                    gout7 += gx[576] * gy[192] * gz[320];
                    gout8 += gx[576] * gy[128] * gz[384];
                    gout9 += gx[448] * gy[640] * gz[0];
                    gout10 += gx[384] * gy[576] * gz[128];
                    gout11 += gx[192] * gy[896] * gz[0];
                    gout12 += gx[256] * gy[640] * gz[192];
                    gout13 += gx[192] * gy[576] * gz[320];
                    gout14 += gx[0] * gy[1088] * gz[0];
                    gout15 += gx[64] * gy[832] * gz[192];
                    gout16 += gx[0] * gy[768] * gz[320];
                    gout17 += gx[0] * gy[704] * gz[384];
                    gout18 += gx[448] * gy[64] * gz[576];
                    gout19 += gx[384] * gy[0] * gz[704];
                    gout20 += gx[192] * gy[320] * gz[576];
                    gout21 += gx[256] * gy[64] * gz[768];
                    gout22 += gx[192] * gy[0] * gz[896];
                    gout23 += gx[0] * gy[512] * gz[576];
                    gout24 += gx[64] * gy[256] * gz[768];
                    gout25 += gx[0] * gy[192] * gz[896];
                    gout26 += gx[0] * gy[128] * gz[960];
                    break;
                    case 2:
                    gout0 += gx[1024] * gy[0] * gz[64];
                    gout1 += gx[896] * gy[192] * gz[0];
                    gout2 += gx[768] * gy[256] * gz[64];
                    gout3 += gx[832] * gy[0] * gz[256];
                    gout4 += gx[704] * gy[384] * gz[0];
                    gout5 += gx[576] * gy[448] * gz[64];
                    gout6 += gx[640] * gy[192] * gz[256];
                    gout7 += gx[704] * gy[0] * gz[384];
                    gout8 += gx[576] * gy[64] * gz[448];
                    gout9 += gx[448] * gy[576] * gz[64];
                    gout10 += gx[320] * gy[768] * gz[0];
                    gout11 += gx[192] * gy[832] * gz[64];
                    gout12 += gx[256] * gy[576] * gz[256];
                    gout13 += gx[128] * gy[960] * gz[0];
                    gout14 += gx[0] * gy[1024] * gz[64];
                    gout15 += gx[64] * gy[768] * gz[256];
                    gout16 += gx[128] * gy[576] * gz[384];
                    gout17 += gx[0] * gy[640] * gz[448];
                    gout18 += gx[448] * gy[0] * gz[640];
                    gout19 += gx[320] * gy[192] * gz[576];
                    gout20 += gx[192] * gy[256] * gz[640];
                    gout21 += gx[256] * gy[0] * gz[832];
                    gout22 += gx[128] * gy[384] * gz[576];
                    gout23 += gx[0] * gy[448] * gz[640];
                    gout24 += gx[64] * gy[192] * gz[832];
                    gout25 += gx[128] * gy[0] * gz[960];
                    gout26 += gx[0] * gy[64] * gz[1024];
                    break;
                    case 3:
                    gout0 += gx[960] * gy[128] * gz[0];
                    gout1 += gx[832] * gy[256] * gz[0];
                    gout2 += gx[768] * gy[192] * gz[128];
                    gout3 += gx[768] * gy[128] * gz[192];
                    gout4 += gx[640] * gy[448] * gz[0];
                    gout5 += gx[576] * gy[384] * gz[128];
                    gout6 += gx[576] * gy[320] * gz[192];
                    gout7 += gx[640] * gy[64] * gz[384];
                    gout8 += gx[576] * gy[0] * gz[512];
                    gout9 += gx[384] * gy[704] * gz[0];
                    gout10 += gx[256] * gy[832] * gz[0];
                    gout11 += gx[192] * gy[768] * gz[128];
                    gout12 += gx[192] * gy[704] * gz[192];
                    gout13 += gx[64] * gy[1024] * gz[0];
                    gout14 += gx[0] * gy[960] * gz[128];
                    gout15 += gx[0] * gy[896] * gz[192];
                    gout16 += gx[64] * gy[640] * gz[384];
                    gout17 += gx[0] * gy[576] * gz[512];
                    gout18 += gx[384] * gy[128] * gz[576];
                    gout19 += gx[256] * gy[256] * gz[576];
                    gout20 += gx[192] * gy[192] * gz[704];
                    gout21 += gx[192] * gy[128] * gz[768];
                    gout22 += gx[64] * gy[448] * gz[576];
                    gout23 += gx[0] * gy[384] * gz[704];
                    gout24 += gx[0] * gy[320] * gz[768];
                    gout25 += gx[64] * gy[64] * gz[960];
                    gout26 += gx[0] * gy[0] * gz[1088];
                    break;
                    }
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*36*n_ctr_pairs + pair_mapping[pair_ij_idx];
            switch (gout_id) {
            case 0:
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+20*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+24*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+28*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+32*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+36*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+40*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+44*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+48*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+52*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+56*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+60*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+64*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+68*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+72*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+76*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+80*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+84*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+88*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+92*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+96*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+100*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+104*n_ctr_pairs, gout26);
            break;
            case 1:
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+21*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+25*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+29*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+33*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+37*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+41*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+45*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+49*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+53*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+57*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+61*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+65*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+69*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+73*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+77*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+81*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+85*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+89*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+93*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+97*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+101*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+105*n_ctr_pairs, gout26);
            break;
            case 2:
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+18*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+22*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+26*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+30*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+34*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+38*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+42*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+46*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+50*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+54*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+58*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+62*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+66*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+70*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+74*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+78*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+82*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+86*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+90*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+94*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+98*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+102*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+106*n_ctr_pairs, gout26);
            break;
            case 3:
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+19*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+23*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+27*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+31*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+35*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+39*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+43*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+47*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+51*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+55*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+59*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+63*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+67*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+71*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+75*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+79*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+83*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+87*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+91*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+95*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+99*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+103*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+107*n_ctr_pairs, gout26);
            break;
            }
        }
    }
}

#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void int3c2e_002(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 16 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[16];

    int ntasks = nksh * 16 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 512) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 2048;
                rys_roots(2, theta_rr, rw1, 512, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 512, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *512] *= theta_fac;
                    rw[(irys*2+1)*512] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*512] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*512];
                    double rt = rw[ 2*irys   *512];
                    double rt_aa = rt / (aij + ak);
                    double rt_ak = rt_aa * aij;
                    double b01 = .5/ak * (1 - rt_ak);
                    double cpx = xpq*rt_ak;
                    double trr_01x = cpx * 1;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    gout0 += trr_02x * 1 * wt;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout1 += trr_01x * trr_01y * wt;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout2 += trr_01x * 1 * trr_01z;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    gout3 += 1 * trr_02y * wt;
                    gout4 += 1 * trr_01y * trr_01z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    gout5 += 1 * 1 * trr_02z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*1*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
        }
    }
}

__global__
void int3c2e_102(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1024;
                rys_roots(2, theta_rr, rw1, 256, 0, 1);
                rys_roots(2, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 2; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
                    double rt_aa = rt / (aij + ak);
                    double b00 = .5 * rt_aa;
                    double rt_ak = rt_aa * aij;
                    double b01 = .5/ak * (1 - rt_ak);
                    double cpx = xpq*rt_ak;
                    double rt_aij = rt_aa * ak;
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    gout0 += trr_12x * 1 * wt;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += trr_02x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_02x * 1 * trr_10z;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout3 += trr_11x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout4 += trr_01x * trr_11y * wt;
                    gout5 += trr_01x * trr_01y * trr_10z;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout6 += trr_11x * 1 * trr_01z;
                    gout7 += trr_01x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout8 += trr_01x * 1 * trr_11z;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    gout9 += trr_10x * trr_02y * wt;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    gout10 += 1 * trr_12y * wt;
                    gout11 += 1 * trr_02y * trr_10z;
                    gout12 += trr_10x * trr_01y * trr_01z;
                    gout13 += 1 * trr_11y * trr_01z;
                    gout14 += 1 * trr_01y * trr_11z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    gout15 += trr_10x * 1 * trr_02z;
                    gout16 += 1 * trr_10y * trr_02z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    gout17 += 1 * 1 * trr_12z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*3*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
        }
    }
}

__global__
void int3c2e_112(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        double gout36 = 0;
        double gout37 = 0;
        double gout38 = 0;
        double gout39 = 0;
        double gout40 = 0;
        double gout41 = 0;
        double gout42 = 0;
        double gout43 = 0;
        double gout44 = 0;
        double gout45 = 0;
        double gout46 = 0;
        double gout47 = 0;
        double gout48 = 0;
        double gout49 = 0;
        double gout50 = 0;
        double gout51 = 0;
        double gout52 = 0;
        double gout53 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1536;
                rys_roots(3, theta_rr, rw1, 256, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 6; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
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
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double hrr_112x = trr_22x - xjxi * trr_12x;
                    gout0 += hrr_112x * 1 * wt;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double hrr_012x = trr_12x - xjxi * trr_02x;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += hrr_012x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_012x * 1 * trr_10z;
                    double hrr_010y = trr_10y - yjyi * 1;
                    gout3 += trr_12x * hrr_010y * wt;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double hrr_110y = trr_20y - yjyi * trr_10y;
                    gout4 += trr_02x * hrr_110y * wt;
                    gout5 += trr_02x * hrr_010y * trr_10z;
                    double hrr_010z = trr_10z - zjzi * wt;
                    gout6 += trr_12x * 1 * hrr_010z;
                    gout7 += trr_02x * trr_10y * hrr_010z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double hrr_110z = trr_20z - zjzi * trr_10z;
                    gout8 += trr_02x * 1 * hrr_110z;
                    double hrr_111x = trr_21x - xjxi * trr_11x;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout9 += hrr_111x * trr_01y * wt;
                    double hrr_011x = trr_11x - xjxi * trr_01x;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout10 += hrr_011x * trr_11y * wt;
                    gout11 += hrr_011x * trr_01y * trr_10z;
                    double hrr_011y = trr_11y - yjyi * trr_01y;
                    gout12 += trr_11x * hrr_011y * wt;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double hrr_111y = trr_21y - yjyi * trr_11y;
                    gout13 += trr_01x * hrr_111y * wt;
                    gout14 += trr_01x * hrr_011y * trr_10z;
                    gout15 += trr_11x * trr_01y * hrr_010z;
                    gout16 += trr_01x * trr_11y * hrr_010z;
                    gout17 += trr_01x * trr_01y * hrr_110z;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout18 += hrr_111x * 1 * trr_01z;
                    gout19 += hrr_011x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout20 += hrr_011x * 1 * trr_11z;
                    gout21 += trr_11x * hrr_010y * trr_01z;
                    gout22 += trr_01x * hrr_110y * trr_01z;
                    gout23 += trr_01x * hrr_010y * trr_11z;
                    double hrr_011z = trr_11z - zjzi * trr_01z;
                    gout24 += trr_11x * 1 * hrr_011z;
                    gout25 += trr_01x * trr_10y * hrr_011z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double hrr_111z = trr_21z - zjzi * trr_11z;
                    gout26 += trr_01x * 1 * hrr_111z;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    gout27 += hrr_110x * trr_02y * wt;
                    double hrr_010x = trr_10x - xjxi * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    gout28 += hrr_010x * trr_12y * wt;
                    gout29 += hrr_010x * trr_02y * trr_10z;
                    double hrr_012y = trr_12y - yjyi * trr_02y;
                    gout30 += trr_10x * hrr_012y * wt;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double hrr_112y = trr_22y - yjyi * trr_12y;
                    gout31 += 1 * hrr_112y * wt;
                    gout32 += 1 * hrr_012y * trr_10z;
                    gout33 += trr_10x * trr_02y * hrr_010z;
                    gout34 += 1 * trr_12y * hrr_010z;
                    gout35 += 1 * trr_02y * hrr_110z;
                    gout36 += hrr_110x * trr_01y * trr_01z;
                    gout37 += hrr_010x * trr_11y * trr_01z;
                    gout38 += hrr_010x * trr_01y * trr_11z;
                    gout39 += trr_10x * hrr_011y * trr_01z;
                    gout40 += 1 * hrr_111y * trr_01z;
                    gout41 += 1 * hrr_011y * trr_11z;
                    gout42 += trr_10x * trr_01y * hrr_011z;
                    gout43 += 1 * trr_11y * hrr_011z;
                    gout44 += 1 * trr_01y * hrr_111z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    gout45 += hrr_110x * 1 * trr_02z;
                    gout46 += hrr_010x * trr_10y * trr_02z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    gout47 += hrr_010x * 1 * trr_12z;
                    gout48 += trr_10x * hrr_010y * trr_02z;
                    gout49 += 1 * hrr_110y * trr_02z;
                    gout50 += 1 * hrr_010y * trr_12z;
                    double hrr_012z = trr_12z - zjzi * trr_02z;
                    gout51 += trr_10x * 1 * hrr_012z;
                    gout52 += 1 * trr_10y * hrr_012z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double hrr_112z = trr_22z - zjzi * trr_12z;
                    gout53 += 1 * 1 * hrr_112z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*9*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+18*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+19*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+20*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+21*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+22*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+23*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+24*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+25*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+26*n_ctr_pairs, gout26);
            atomicAdd(eri_tensor+27*n_ctr_pairs, gout27);
            atomicAdd(eri_tensor+28*n_ctr_pairs, gout28);
            atomicAdd(eri_tensor+29*n_ctr_pairs, gout29);
            atomicAdd(eri_tensor+30*n_ctr_pairs, gout30);
            atomicAdd(eri_tensor+31*n_ctr_pairs, gout31);
            atomicAdd(eri_tensor+32*n_ctr_pairs, gout32);
            atomicAdd(eri_tensor+33*n_ctr_pairs, gout33);
            atomicAdd(eri_tensor+34*n_ctr_pairs, gout34);
            atomicAdd(eri_tensor+35*n_ctr_pairs, gout35);
            atomicAdd(eri_tensor+36*n_ctr_pairs, gout36);
            atomicAdd(eri_tensor+37*n_ctr_pairs, gout37);
            atomicAdd(eri_tensor+38*n_ctr_pairs, gout38);
            atomicAdd(eri_tensor+39*n_ctr_pairs, gout39);
            atomicAdd(eri_tensor+40*n_ctr_pairs, gout40);
            atomicAdd(eri_tensor+41*n_ctr_pairs, gout41);
            atomicAdd(eri_tensor+42*n_ctr_pairs, gout42);
            atomicAdd(eri_tensor+43*n_ctr_pairs, gout43);
            atomicAdd(eri_tensor+44*n_ctr_pairs, gout44);
            atomicAdd(eri_tensor+45*n_ctr_pairs, gout45);
            atomicAdd(eri_tensor+46*n_ctr_pairs, gout46);
            atomicAdd(eri_tensor+47*n_ctr_pairs, gout47);
            atomicAdd(eri_tensor+48*n_ctr_pairs, gout48);
            atomicAdd(eri_tensor+49*n_ctr_pairs, gout49);
            atomicAdd(eri_tensor+50*n_ctr_pairs, gout50);
            atomicAdd(eri_tensor+51*n_ctr_pairs, gout51);
            atomicAdd(eri_tensor+52*n_ctr_pairs, gout52);
            atomicAdd(eri_tensor+53*n_ctr_pairs, gout53);
        }
    }
}

__global__
void int3c2e_202(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = threadIdx.z * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 8 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    __shared__ int img_counts_in_warp[8];

    int ntasks = nksh * 8 * SPTAKS_PER_BLOCK;
    for (int ijk_idx = ksp_id; ijk_idx < ntasks; ijk_idx += 256) {
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double gout27 = 0;
        double gout28 = 0;
        double gout29 = 0;
        double gout30 = 0;
        double gout31 = 0;
        double gout32 = 0;
        double gout33 = 0;
        double gout34 = 0;
        double gout35 = 0;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                cicj = 0.;
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
            double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
            double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
            double xpq = xij - rk[0];
            double ypq = yij - rk[1];
            double zpq = zij - rk[2];
            double rr = xpq * xpq + ypq * ypq + zpq * zpq;

            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                double fac = fac_ij * ck[kp] / (aij*ak*sqrt(aij+ak));
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * rr;
                double *rw1 = rw + 1536;
                rys_roots(3, theta_rr, rw1, 256, 0, 1);
                rys_roots(3, theta_fac*theta_rr, rw, 256, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                for (int irys = 0; irys < 3; ++irys) {
                    rw[ irys*2   *256] *= theta_fac;
                    rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    rw1[(irys*2+1)*256] *= fac;
                }
                for (int irys = 0; irys < 6; ++irys) {
                    double wt = rw[(2*irys+1)*256];
                    double rt = rw[ 2*irys   *256];
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
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    gout0 += trr_22x * 1 * wt;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout1 += trr_12x * trr_10y * wt;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_12x * 1 * trr_10z;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout3 += trr_02x * trr_20y * wt;
                    gout4 += trr_02x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += trr_02x * 1 * trr_20z;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * 1;
                    gout6 += trr_21x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout7 += trr_11x * trr_11y * wt;
                    gout8 += trr_11x * trr_01y * trr_10z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout9 += trr_01x * trr_21y * wt;
                    gout10 += trr_01x * trr_11y * trr_10z;
                    gout11 += trr_01x * trr_01y * trr_20z;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    gout12 += trr_21x * 1 * trr_01z;
                    gout13 += trr_11x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout14 += trr_11x * 1 * trr_11z;
                    gout15 += trr_01x * trr_20y * trr_01z;
                    gout16 += trr_01x * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout17 += trr_01x * 1 * trr_21z;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    gout18 += trr_20x * trr_02y * wt;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    gout19 += trr_10x * trr_12y * wt;
                    gout20 += trr_10x * trr_02y * trr_10z;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    gout21 += 1 * trr_22y * wt;
                    gout22 += 1 * trr_12y * trr_10z;
                    gout23 += 1 * trr_02y * trr_20z;
                    gout24 += trr_20x * trr_01y * trr_01z;
                    gout25 += trr_10x * trr_11y * trr_01z;
                    gout26 += trr_10x * trr_01y * trr_11z;
                    gout27 += 1 * trr_21y * trr_01z;
                    gout28 += 1 * trr_11y * trr_11z;
                    gout29 += 1 * trr_01y * trr_21z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    gout30 += trr_20x * 1 * trr_02z;
                    gout31 += trr_10x * trr_10y * trr_02z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    gout32 += trr_10x * 1 * trr_12z;
                    gout33 += 1 * trr_20y * trr_02z;
                    gout34 += 1 * trr_10y * trr_12z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    gout35 += 1 * 1 * trr_22z;
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*6*n_ctr_pairs + pair_mapping[pair_ij_idx];
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+18*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+19*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+20*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+21*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+22*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+23*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+24*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+25*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+26*n_ctr_pairs, gout26);
            atomicAdd(eri_tensor+27*n_ctr_pairs, gout27);
            atomicAdd(eri_tensor+28*n_ctr_pairs, gout28);
            atomicAdd(eri_tensor+29*n_ctr_pairs, gout29);
            atomicAdd(eri_tensor+30*n_ctr_pairs, gout30);
            atomicAdd(eri_tensor+31*n_ctr_pairs, gout31);
            atomicAdd(eri_tensor+32*n_ctr_pairs, gout32);
            atomicAdd(eri_tensor+33*n_ctr_pairs, gout33);
            atomicAdd(eri_tensor+34*n_ctr_pairs, gout34);
            atomicAdd(eri_tensor+35*n_ctr_pairs, gout35);
        }
    }
}

__global__
void int3c2e_212(double *out, PBCInt3c2eEnvVars envs, PBCInt3c2eBounds bounds)
{
    int ksh_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int sp_id = threadIdx.z;
    int sp_block_id = blockIdx.x;
    int ksh_block_id = blockIdx.y;
    int ksp_id = 32 * sp_id + ksh_id;
    int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int nimgs = envs.nimgs;
    int sp0_this_block = sp_block_id * 2 * SPTAKS_PER_BLOCK;
    int ksh0_this_block = ksh_block_id * 32;
    int nksh = MIN(bounds.nksh - ksh0_this_block, 32);
    int ksh0 = ksh0_this_block + bounds.ksh0;
    int kprim = bounds.kprim;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int *img_idx = bounds.img_idx;
    int *sp_img_offsets = bounds.img_offsets;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + ksp_id;
    double *gx = rw + 768;
    double *gy = gx + 1152;
    double *gz = gy + 1152;
    double *rjri = gz + 1152;
    double *Rpq = rjri + 192;
    __shared__ int img_counts_in_warp[WARPS];

    int ntasks = nksh * 2 * SPTAKS_PER_BLOCK;
    for (int task0 = 0; task0 < ntasks; task0 += 64) {
        int ijk_idx = task0 + ksp_id;
        int ksh = ijk_idx % nksh + ksh0;
        int pair_ij_idx = ijk_idx / nksh + sp0_this_block;
        int img1 = 1;
        int pair_ij = pair_ij_idx;
        if (pair_ij_idx >= bounds.n_prim_pairs) {
            pair_ij = sp0_this_block;
        } else {
            img1 = sp_img_offsets[pair_ij_idx+1];
        }
        int bas_ij = bounds.bas_ij_idx[pair_ij];
        int img0 = sp_img_offsets[pair_ij];
        int thread_id_in_warp = thread_id % WARP_SIZE;
        if (thread_id_in_warp == 0) {
            img_counts_in_warp[warp_id] = 0;
        }
        atomicMax(&img_counts_in_warp[warp_id], img1-img0);
        __syncthreads();
        gy[0] = 1.;

        int nbas = envs.cell0_nbas * envs.bvk_ncells;
        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
        double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
        double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
        double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
        double aij = ai + aj;
        double cicj = ci * cj;
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double gout0 = 0;
        double gout1 = 0;
        double gout2 = 0;
        double gout3 = 0;
        double gout4 = 0;
        double gout5 = 0;
        double gout6 = 0;
        double gout7 = 0;
        double gout8 = 0;
        double gout9 = 0;
        double gout10 = 0;
        double gout11 = 0;
        double gout12 = 0;
        double gout13 = 0;
        double gout14 = 0;
        double gout15 = 0;
        double gout16 = 0;
        double gout17 = 0;
        double gout18 = 0;
        double gout19 = 0;
        double gout20 = 0;
        double gout21 = 0;
        double gout22 = 0;
        double gout23 = 0;
        double gout24 = 0;
        double gout25 = 0;
        double gout26 = 0;
        double s0, s1, s2;
        int img_counts = img_counts_in_warp[warp_id];
        for (int img = 0; img < img_counts; ++img) {
            int img_id = img0 + img;
            __syncthreads();
            if (img_id >= img1) {
                // ensure the same number of images processed in the same warp
                img_id = img0;
                if (gout_id == 0) {
                    gy[0] = 0.;
                }
            }
            int img_ij = img_idx[img_id];
            int iL = img_ij / nimgs;
            int jL = img_ij % nimgs;
            double xi = ri[0];
            double yi = ri[1];
            double zi = ri[2];
            double xj = rj[0];
            double yj = rj[1];
            double zj = rj[2];
            double xjxi = xj + img_coords[jL*3+0] - xi;
            double yjyi = yj + img_coords[jL*3+1] - yi;
            double zjzi = zj + img_coords[jL*3+2] - zi;
            double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
            double aj_aij = aj / aij;
            double theta_ij = ai * aj_aij;
            double Kab = theta_ij * rr_ij;
            double fac_ij = PI_FAC * cicj * exp(-Kab);
            if (gout_id == 0) {
                double xij = xjxi * aj_aij + xi + img_coords[iL*3+0];
                double yij = yjyi * aj_aij + yi + img_coords[iL*3+1];
                double zij = zjzi * aj_aij + zi + img_coords[iL*3+2];
                double xpq = xij - rk[0];
                double ypq = yij - rk[1];
                double zpq = zij - rk[2];
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rjri[0] = xjxi;
                rjri[64] = yjyi;
                rjri[128] = zjzi;
                Rpq[0] = xpq;
                Rpq[64] = ypq;
                Rpq[128] = zpq;
                Rpq[192] = rr;
            }
            for (int kp = 0; kp < kprim; ++kp) {
                double ak = expk[kp];
                double theta = aij * ak / (aij + ak);
                __syncthreads();
                if (gout_id == 0) {
                    double cijk = fac_ij * ck[kp];
                    gx[0] = cijk / (aij*ak*sqrt(aij+ak));
                }
                double omega2 = omega * omega;
                double theta_fac = omega2 / (omega2 + theta);
                double theta_rr = theta * Rpq[192];
                double *rw1 = rw + 384;
                __syncthreads();
                rys_roots(3, theta_rr, rw1, 64, gout_id, 4);
                rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                __syncthreads();
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = gout_id; irys < 3; irys += 4) {
                    rw[ irys*2   *64] *= theta_fac;
                    rw[(irys*2+1)*64] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 6; ++irys) {
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
                    gout0 += gx[1088] * gy[0] * gz[0];
                    gout1 += gx[960] * gy[64] * gz[64];
                    gout2 += gx[832] * gy[192] * gz[64];
                    gout3 += gx[896] * gy[0] * gz[192];
                    gout4 += gx[768] * gy[64] * gz[256];
                    gout5 += gx[640] * gy[384] * gz[64];
                    gout6 += gx[512] * gy[576] * gz[0];
                    gout7 += gx[384] * gy[640] * gz[64];
                    gout8 += gx[448] * gy[384] * gz[256];
                    gout9 += gx[704] * gy[0] * gz[384];
                    gout10 += gx[576] * gy[64] * gz[448];
                    gout11 += gx[448] * gy[192] * gz[448];
                    gout12 += gx[512] * gy[0] * gz[576];
                    gout13 += gx[384] * gy[64] * gz[640];
                    gout14 += gx[256] * gy[768] * gz[64];
                    gout15 += gx[128] * gy[960] * gz[0];
                    gout16 += gx[0] * gy[1024] * gz[64];
                    gout17 += gx[64] * gy[768] * gz[256];
                    gout18 += gx[320] * gy[384] * gz[384];
                    gout19 += gx[192] * gy[448] * gz[448];
                    gout20 += gx[64] * gy[576] * gz[448];
                    gout21 += gx[128] * gy[384] * gz[576];
                    gout22 += gx[0] * gy[448] * gz[640];
                    gout23 += gx[256] * gy[0] * gz[832];
                    gout24 += gx[128] * gy[192] * gz[768];
                    gout25 += gx[0] * gy[256] * gz[832];
                    gout26 += gx[64] * gy[0] * gz[1024];
                    break;
                    case 1:
                    gout0 += gx[1024] * gy[64] * gz[0];
                    gout1 += gx[960] * gy[0] * gz[128];
                    gout2 += gx[768] * gy[320] * gz[0];
                    gout3 += gx[832] * gy[64] * gz[192];
                    gout4 += gx[768] * gy[0] * gz[320];
                    gout5 += gx[576] * gy[512] * gz[0];
                    gout6 += gx[448] * gy[640] * gz[0];
                    gout7 += gx[384] * gy[576] * gz[128];
                    gout8 += gx[384] * gy[512] * gz[192];
                    gout9 += gx[640] * gy[64] * gz[384];
                    gout10 += gx[576] * gy[0] * gz[512];
                    gout11 += gx[384] * gy[320] * gz[384];
                    gout12 += gx[448] * gy[64] * gz[576];
                    gout13 += gx[384] * gy[0] * gz[704];
                    gout14 += gx[192] * gy[896] * gz[0];
                    gout15 += gx[64] * gy[1024] * gz[0];
                    gout16 += gx[0] * gy[960] * gz[128];
                    gout17 += gx[0] * gy[896] * gz[192];
                    gout18 += gx[256] * gy[448] * gz[384];
                    gout19 += gx[192] * gy[384] * gz[512];
                    gout20 += gx[0] * gy[704] * gz[384];
                    gout21 += gx[64] * gy[448] * gz[576];
                    gout22 += gx[0] * gy[384] * gz[704];
                    gout23 += gx[192] * gy[128] * gz[768];
                    gout24 += gx[64] * gy[256] * gz[768];
                    gout25 += gx[0] * gy[192] * gz[896];
                    gout26 += gx[0] * gy[128] * gz[960];
                    break;
                    case 2:
                    gout0 += gx[1024] * gy[0] * gz[64];
                    gout1 += gx[896] * gy[192] * gz[0];
                    gout2 += gx[768] * gy[256] * gz[64];
                    gout3 += gx[832] * gy[0] * gz[256];
                    gout4 += gx[704] * gy[384] * gz[0];
                    gout5 += gx[576] * gy[448] * gz[64];
                    gout6 += gx[448] * gy[576] * gz[64];
                    gout7 += gx[512] * gy[384] * gz[192];
                    gout8 += gx[384] * gy[448] * gz[256];
                    gout9 += gx[640] * gy[0] * gz[448];
                    gout10 += gx[512] * gy[192] * gz[384];
                    gout11 += gx[384] * gy[256] * gz[448];
                    gout12 += gx[448] * gy[0] * gz[640];
                    gout13 += gx[320] * gy[768] * gz[0];
                    gout14 += gx[192] * gy[832] * gz[64];
                    gout15 += gx[64] * gy[960] * gz[64];
                    gout16 += gx[128] * gy[768] * gz[192];
                    gout17 += gx[0] * gy[832] * gz[256];
                    gout18 += gx[256] * gy[384] * gz[448];
                    gout19 += gx[128] * gy[576] * gz[384];
                    gout20 += gx[0] * gy[640] * gz[448];
                    gout21 += gx[64] * gy[384] * gz[640];
                    gout22 += gx[320] * gy[0] * gz[768];
                    gout23 += gx[192] * gy[64] * gz[832];
                    gout24 += gx[64] * gy[192] * gz[832];
                    gout25 += gx[128] * gy[0] * gz[960];
                    gout26 += gx[0] * gy[64] * gz[1024];
                    break;
                    case 3:
                    gout0 += gx[960] * gy[128] * gz[0];
                    gout1 += gx[832] * gy[256] * gz[0];
                    gout2 += gx[768] * gy[192] * gz[128];
                    gout3 += gx[768] * gy[128] * gz[192];
                    gout4 += gx[640] * gy[448] * gz[0];
                    gout5 += gx[576] * gy[384] * gz[128];
                    gout6 += gx[384] * gy[704] * gz[0];
                    gout7 += gx[448] * gy[448] * gz[192];
                    gout8 += gx[384] * gy[384] * gz[320];
                    gout9 += gx[576] * gy[128] * gz[384];
                    gout10 += gx[448] * gy[256] * gz[384];
                    gout11 += gx[384] * gy[192] * gz[512];
                    gout12 += gx[384] * gy[128] * gz[576];
                    gout13 += gx[256] * gy[832] * gz[0];
                    gout14 += gx[192] * gy[768] * gz[128];
                    gout15 += gx[0] * gy[1088] * gz[0];
                    gout16 += gx[64] * gy[832] * gz[192];
                    gout17 += gx[0] * gy[768] * gz[320];
                    gout18 += gx[192] * gy[512] * gz[384];
                    gout19 += gx[64] * gy[640] * gz[384];
                    gout20 += gx[0] * gy[576] * gz[512];
                    gout21 += gx[0] * gy[512] * gz[576];
                    gout22 += gx[256] * gy[64] * gz[768];
                    gout23 += gx[192] * gy[0] * gz[896];
                    gout24 += gx[0] * gy[320] * gz[768];
                    gout25 += gx[64] * gy[64] * gz[960];
                    gout26 += gx[0] * gy[0] * gz[1088];
                    break;
                    }
                }
            }
        }
        if (pair_ij_idx < bounds.n_prim_pairs) {
            int *ao_loc = envs.ao_loc;
            int *pair_mapping = bounds.pair_mapping;
            size_t n_ctr_pairs = bounds.n_ctr_pairs;
            int k0 = ao_loc[ksh] - ao_loc[bounds.ksh0];
            double *eri_tensor = out + k0*18*n_ctr_pairs + pair_mapping[pair_ij_idx];
            switch (gout_id) {
            case 0:
            atomicAdd(eri_tensor+0*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+4*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+8*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+12*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+16*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+20*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+24*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+28*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+32*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+36*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+40*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+44*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+48*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+52*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+56*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+60*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+64*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+68*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+72*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+76*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+80*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+84*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+88*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+92*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+96*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+100*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+104*n_ctr_pairs, gout26);
            break;
            case 1:
            atomicAdd(eri_tensor+1*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+5*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+9*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+13*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+17*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+21*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+25*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+29*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+33*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+37*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+41*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+45*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+49*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+53*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+57*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+61*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+65*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+69*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+73*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+77*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+81*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+85*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+89*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+93*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+97*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+101*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+105*n_ctr_pairs, gout26);
            break;
            case 2:
            atomicAdd(eri_tensor+2*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+6*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+10*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+14*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+18*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+22*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+26*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+30*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+34*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+38*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+42*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+46*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+50*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+54*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+58*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+62*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+66*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+70*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+74*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+78*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+82*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+86*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+90*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+94*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+98*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+102*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+106*n_ctr_pairs, gout26);
            break;
            case 3:
            atomicAdd(eri_tensor+3*n_ctr_pairs, gout0);
            atomicAdd(eri_tensor+7*n_ctr_pairs, gout1);
            atomicAdd(eri_tensor+11*n_ctr_pairs, gout2);
            atomicAdd(eri_tensor+15*n_ctr_pairs, gout3);
            atomicAdd(eri_tensor+19*n_ctr_pairs, gout4);
            atomicAdd(eri_tensor+23*n_ctr_pairs, gout5);
            atomicAdd(eri_tensor+27*n_ctr_pairs, gout6);
            atomicAdd(eri_tensor+31*n_ctr_pairs, gout7);
            atomicAdd(eri_tensor+35*n_ctr_pairs, gout8);
            atomicAdd(eri_tensor+39*n_ctr_pairs, gout9);
            atomicAdd(eri_tensor+43*n_ctr_pairs, gout10);
            atomicAdd(eri_tensor+47*n_ctr_pairs, gout11);
            atomicAdd(eri_tensor+51*n_ctr_pairs, gout12);
            atomicAdd(eri_tensor+55*n_ctr_pairs, gout13);
            atomicAdd(eri_tensor+59*n_ctr_pairs, gout14);
            atomicAdd(eri_tensor+63*n_ctr_pairs, gout15);
            atomicAdd(eri_tensor+67*n_ctr_pairs, gout16);
            atomicAdd(eri_tensor+71*n_ctr_pairs, gout17);
            atomicAdd(eri_tensor+75*n_ctr_pairs, gout18);
            atomicAdd(eri_tensor+79*n_ctr_pairs, gout19);
            atomicAdd(eri_tensor+83*n_ctr_pairs, gout20);
            atomicAdd(eri_tensor+87*n_ctr_pairs, gout21);
            atomicAdd(eri_tensor+91*n_ctr_pairs, gout22);
            atomicAdd(eri_tensor+95*n_ctr_pairs, gout23);
            atomicAdd(eri_tensor+99*n_ctr_pairs, gout24);
            atomicAdd(eri_tensor+103*n_ctr_pairs, gout25);
            atomicAdd(eri_tensor+107*n_ctr_pairs, gout26);
            break;
            }
        }
    }
}

int int3c2e_unrolled(double *out, PBCInt3c2eEnvVars *envs, PBCInt3c2eBounds *bounds)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int kij = lk*25 + li*5 + lj;
    int nroots = bounds->nroots;
    int n_prim_pairs = bounds->n_prim_pairs;
    int nksh = bounds->nksh;
    int nksh_per_block = 32;
    int nsp_per_block = 8;
    int gout_stride = 1;

    switch (kij) {
    case 37:
        nksh_per_block = 32;
        nsp_per_block = 2;
        gout_stride = 4;
        break;
    case 61:
        nksh_per_block = 32;
        nsp_per_block = 2;
        gout_stride = 4;
        break;
    }

#if CUDA_VERSION >= 12040
    switch (kij) {
    case 0: nsp_per_block *= 2; break;
    case 5: nsp_per_block *= 2; break;
    case 10: nsp_per_block *= 2; break;
    case 25: nsp_per_block *= 2; break;
    case 50: nsp_per_block *= 2; break;
    }
#endif

    dim3 threads(nksh_per_block, gout_stride, nsp_per_block);
    int sp_blocks = (n_prim_pairs + SPTAKS_PER_BLOCK*nsp_per_block - 1) /
        (SPTAKS_PER_BLOCK*nsp_per_block);
    int ksh_blocks = (nksh + nksh_per_block - 1) / nksh_per_block;
    dim3 blocks(sp_blocks, ksh_blocks);
    int buflen = nroots*2 * nksh_per_block * nsp_per_block;
    switch (kij) {
    case 0:
        int3c2e_000<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 5:
        int3c2e_100<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 6:
        int3c2e_110<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 10:
        int3c2e_200<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 11:
        int3c2e_210<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 12:
        int3c2e_220<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 25:
        int3c2e_001<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 30:
        int3c2e_101<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 31:
        int3c2e_111<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 35:
        int3c2e_201<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 36:
        int3c2e_211<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 37:
        buflen += 3904;
        int3c2e_221<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 50:
        int3c2e_002<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 55:
        int3c2e_102<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 56:
        int3c2e_112<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 60:
        int3c2e_202<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    case 61:
        buflen += 3904;
        int3c2e_212<<<blocks, threads, buflen*sizeof(double)>>>(out, *envs, *bounds); break;
    default: return 0;
    }
    return 1;
}
