#include <cuda.h>
#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks_ip1.cu"


__device__ static
void _rys_vjk_ip1_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + sq_id;
    double *gx = rw + 512 * nroots;
    double *gy = gx + 512;
    double *gz = gy + 512;
    double *rjri_cache = rw_cache + 256 * (nroots*2 + 6);
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    double s0, s1, s2;
    double Ix, Iy, Iz;

    __syncthreads();
    ShellQuartet sq = shl_quartet_idx[0];
    int ish = sq.i;
    int jsh = sq.j;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double xjxi = rj[0] - ri[0];
    double yjyi = rj[1] - ri[1];
    double zjzi = rj[2] - ri[2];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double *rjri = rjri_cache + ij*6;
        rjri[0] = xjxi;
        rjri[1] = yjyi;
        rjri[2] = zjzi;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        rjri[3] = ci[ip] * cj[jp] * Kab;
        rjri[4] = aij;
        rjri[5] = ai * 2;
    }
    double gout0x, gout0y, gout0z;
    double ai2, fx, fy, fz;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        if (ksh == lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
            gout0x = 0;
            gout0y = 0;
            gout0z = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = expk[kp];
                double al = expl[lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    double *rjri = rjri_cache + ijp*6;
                    double aij = rjri[4];
                    double ai = rjri[5] * .5;
                    double aj_aij = 1. - ai / aij;
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = rk[0] + xqc;
                    double ykl = rk[1] + yqc;
                    double zkl = rk[2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    Rpq[0] = xpq;
                    Rpq[1] = ypq;
                    Rpq[2] = zpq;
                    double cicj = rjri[3];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    double theta_rr = theta * rr;
                    __syncthreads();
                    if (omega == 0) {
                        rys_roots(1, theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[(irys*2+1)*256] *= fac;
                        }
                    } else if (omega > 0) {
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                        }
                    } else {
                        double *rw1 = rw + 512;
                        rys_roots(1, theta_rr, rw1, 256, gout_id, gout_stride);
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                            rw1[(irys*2+1)*256] *= fac;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double rt = rw[irys*512];
                        double aij = rjri[4];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        for (int n = gout_id; n < 3; n += 1) {
                            if (n == 2) {
                                gz[0] = rw[irys*512+256];
                            }
                            double *_gx = gx + n * 512;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aj_aij;
                            double c0x = Rpa - rt_aij * Rpq[n];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[256] = s1;
                        }
                        __syncthreads();
                        ai2 = rjri[5];
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[0];
                        gout0x += ai2 * gx[256] * Iy * Iz;
                        gout0y += ai2 * gy[256] * Ix * Iz;
                        gout0z += ai2 * gz[256] * Ix * Iy;
                    }
                }
            }
            if (task_id < ntasks) {
                int ia = bas[ish*BAS_SLOTS+ATOM_OF] - jk.atom_offset;
                double *dm = jk.dm;
                int do_j = jk.vj != NULL;
                int do_k = jk.vk != NULL;
                double *vj_x = jk.vj + (ia*3+0)*nao*nao;
                double *vj_y = jk.vj + (ia*3+1)*nao*nao;
                double *vj_z = jk.vj + (ia*3+2)*nao*nao;
                double *vk_x = jk.vk + (ia*3+0)*nao*nao;
                double *vk_y = jk.vk + (ia*3+1)*nao*nao;
                double *vk_z = jk.vk + (ia*3+2)*nao*nao;
                if (do_j) {
                    fx = gout0x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj_x+(i0+0)*nao+(j0+0), fx);
                    atomicAdd(vj_y+(i0+0)*nao+(j0+0), fy);
                    atomicAdd(vj_z+(i0+0)*nao+(j0+0), fz);
                    fx = gout0x * dm[(j0+0)*nao+(i0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(i0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(i0+0)];
                    atomicAdd(vj_x+(k0+0)*nao+(l0+0), fx);
                    atomicAdd(vj_y+(k0+0)*nao+(l0+0), fy);
                    atomicAdd(vj_z+(k0+0)*nao+(l0+0), fz);
                }
                if (do_k) {
                    fx = gout0x * dm[(j0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(l0+0), fz);
                    fx = gout0x * dm[(i0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(l0+0), fz);
                    fx = gout0x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(k0+0), fz);
                    fx = gout0x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(k0+0), fz);
                }
            }
    }
}
__global__
void rys_vjk_ip1_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH1;
    extern __shared__ int batch_id[];
    if (t_id == 0) {
        batch_id[0] = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.npairs_kl + QUEUE_DEPTH1 - 1) / QUEUE_DEPTH1;
    int nbatches = bounds.npairs_ij * nbatches_kl;
    while (batch_id[0] < nbatches) {
        int batch_ij = batch_id[0] / nbatches_kl;
        int batch_kl = batch_id[0] % nbatches_kl;
        int ntasks = _fill_jk_tasks_s2kl(shl_quartet_idx, envs, jk, bounds,
                                         batch_ij, batch_kl);
        if (ntasks > 0) {
            _rys_vjk_ip1_0000(envs, jk, bounds, shl_quartet_idx, ntasks);
            __syncthreads();
        }
        if (t_id == 0) {
            batch_id[0] = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_vjk_ip1_0010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + sq_id;
    double *gx = rw + 512 * nroots;
    double *gy = gx + 1024;
    double *gz = gy + 1024;
    double *rjri_cache = rw_cache + 256 * (nroots*2 + 12);
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    double s0, s1, s2;
    double Ix, Iy, Iz;

    __syncthreads();
    ShellQuartet sq = shl_quartet_idx[0];
    int ish = sq.i;
    int jsh = sq.j;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double xjxi = rj[0] - ri[0];
    double yjyi = rj[1] - ri[1];
    double zjzi = rj[2] - ri[2];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double *rjri = rjri_cache + ij*6;
        rjri[0] = xjxi;
        rjri[1] = yjyi;
        rjri[2] = zjzi;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        rjri[3] = ci[ip] * cj[jp] * Kab;
        rjri[4] = aij;
        rjri[5] = ai * 2;
    }
    double gout0x, gout0y, gout0z;
    double gout1x, gout1y, gout1z;
    double gout2x, gout2y, gout2z;
    double ai2, fx, fy, fz;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        if (ksh == lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
            gout0x = 0;
            gout0y = 0;
            gout0z = 0;
            gout1x = 0;
            gout1y = 0;
            gout1z = 0;
            gout2x = 0;
            gout2y = 0;
            gout2z = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = expk[kp];
                double al = expl[lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    double *rjri = rjri_cache + ijp*6;
                    double aij = rjri[4];
                    double ai = rjri[5] * .5;
                    double aj_aij = 1. - ai / aij;
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = rk[0] + xqc;
                    double ykl = rk[1] + yqc;
                    double zkl = rk[2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    Rpq[0] = xpq;
                    Rpq[1] = ypq;
                    Rpq[2] = zpq;
                    double cicj = rjri[3];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    double theta_rr = theta * rr;
                    __syncthreads();
                    if (omega == 0) {
                        rys_roots(1, theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[(irys*2+1)*256] *= fac;
                        }
                    } else if (omega > 0) {
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                        }
                    } else {
                        double *rw1 = rw + 512;
                        rys_roots(1, theta_rr, rw1, 256, gout_id, gout_stride);
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                            rw1[(irys*2+1)*256] *= fac;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double rt = rw[irys*512];
                        double aij = rjri[4];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        for (int n = gout_id; n < 3; n += 1) {
                            if (n == 2) {
                                gz[0] = rw[irys*512+256];
                            }
                            double *_gx = gx + n * 1024;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aj_aij;
                            double c0x = Rpa - rt_aij * Rpq[n];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[256] = s1;
                            double xlxk = rl[n] - rk[n];
                            double Rqc = xlxk * al_akl;
                            double cpx = Rqc + rt_akl * Rpq[n];
                            s0 = _gx[0];
                            s1 = cpx * s0;
                            _gx[512] = s1;
                            s0 = _gx[256];
                            s1 = cpx * s0;
                            s1 += 1 * b00 * _gx[0];
                            _gx[768] = s1;
                        }
                        __syncthreads();
                        ai2 = rjri[5];
                        Ix = gx[512];
                        Iy = gy[0];
                        Iz = gz[0];
                        gout0x += ai2 * gx[768] * Iy * Iz;
                        gout0y += ai2 * gy[256] * Ix * Iz;
                        gout0z += ai2 * gz[256] * Ix * Iy;
                        ai2 = rjri[5];
                        Ix = gx[0];
                        Iy = gy[512];
                        Iz = gz[0];
                        gout1x += ai2 * gx[256] * Iy * Iz;
                        gout1y += ai2 * gy[768] * Ix * Iz;
                        gout1z += ai2 * gz[256] * Ix * Iy;
                        ai2 = rjri[5];
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[512];
                        gout2x += ai2 * gx[256] * Iy * Iz;
                        gout2y += ai2 * gy[256] * Ix * Iz;
                        gout2z += ai2 * gz[768] * Ix * Iy;
                    }
                }
            }
            if (task_id < ntasks) {
                int ia = bas[ish*BAS_SLOTS+ATOM_OF] - jk.atom_offset;
                double *dm = jk.dm;
                int do_j = jk.vj != NULL;
                int do_k = jk.vk != NULL;
                double *vj_x = jk.vj + (ia*3+0)*nao*nao;
                double *vj_y = jk.vj + (ia*3+1)*nao*nao;
                double *vj_z = jk.vj + (ia*3+2)*nao*nao;
                double *vk_x = jk.vk + (ia*3+0)*nao*nao;
                double *vk_y = jk.vk + (ia*3+1)*nao*nao;
                double *vk_z = jk.vk + (ia*3+2)*nao*nao;
                if (do_j) {
                    fx = gout0x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(l0+0)*nao+(k0+0)];
                    fx += gout1x * dm[(l0+0)*nao+(k0+1)];
                    fy += gout1y * dm[(l0+0)*nao+(k0+1)];
                    fz += gout1z * dm[(l0+0)*nao+(k0+1)];
                    fx += gout2x * dm[(l0+0)*nao+(k0+2)];
                    fy += gout2y * dm[(l0+0)*nao+(k0+2)];
                    fz += gout2z * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj_x+(i0+0)*nao+(j0+0), fx);
                    atomicAdd(vj_y+(i0+0)*nao+(j0+0), fy);
                    atomicAdd(vj_z+(i0+0)*nao+(j0+0), fz);
                    fx = gout0x * dm[(j0+0)*nao+(i0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(i0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(i0+0)];
                    atomicAdd(vj_x+(k0+0)*nao+(l0+0), fx);
                    atomicAdd(vj_y+(k0+0)*nao+(l0+0), fy);
                    atomicAdd(vj_z+(k0+0)*nao+(l0+0), fz);
                    fx = gout1x * dm[(j0+0)*nao+(i0+0)];
                    fy = gout1y * dm[(j0+0)*nao+(i0+0)];
                    fz = gout1z * dm[(j0+0)*nao+(i0+0)];
                    atomicAdd(vj_x+(k0+1)*nao+(l0+0), fx);
                    atomicAdd(vj_y+(k0+1)*nao+(l0+0), fy);
                    atomicAdd(vj_z+(k0+1)*nao+(l0+0), fz);
                    fx = gout2x * dm[(j0+0)*nao+(i0+0)];
                    fy = gout2y * dm[(j0+0)*nao+(i0+0)];
                    fz = gout2z * dm[(j0+0)*nao+(i0+0)];
                    atomicAdd(vj_x+(k0+2)*nao+(l0+0), fx);
                    atomicAdd(vj_y+(k0+2)*nao+(l0+0), fy);
                    atomicAdd(vj_z+(k0+2)*nao+(l0+0), fz);
                }
                if (do_k) {
                    fx = gout0x * dm[(j0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(k0+0)];
                    fx += gout1x * dm[(j0+0)*nao+(k0+1)];
                    fy += gout1y * dm[(j0+0)*nao+(k0+1)];
                    fz += gout1z * dm[(j0+0)*nao+(k0+1)];
                    fx += gout2x * dm[(j0+0)*nao+(k0+2)];
                    fy += gout2y * dm[(j0+0)*nao+(k0+2)];
                    fz += gout2z * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk_x+(i0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(l0+0), fz);
                    fx = gout0x * dm[(i0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(k0+0)];
                    fx += gout1x * dm[(i0+0)*nao+(k0+1)];
                    fy += gout1y * dm[(i0+0)*nao+(k0+1)];
                    fz += gout1z * dm[(i0+0)*nao+(k0+1)];
                    fx += gout2x * dm[(i0+0)*nao+(k0+2)];
                    fy += gout2y * dm[(i0+0)*nao+(k0+2)];
                    fz += gout2z * dm[(i0+0)*nao+(k0+2)];
                    atomicAdd(vk_x+(j0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(l0+0), fz);
                    fx = gout0x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(k0+0), fz);
                    fx = gout1x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout1y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout1z * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(k0+1), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(k0+1), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(k0+1), fz);
                    fx = gout2x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout2y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout2z * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(k0+2), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(k0+2), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(k0+2), fz);
                    fx = gout0x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(k0+0), fz);
                    fx = gout1x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout1y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout1z * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(k0+1), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(k0+1), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(k0+1), fz);
                    fx = gout2x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout2y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout2z * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(k0+2), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(k0+2), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(k0+2), fz);
                }
            }
    }
}
__global__
void rys_vjk_ip1_0010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH1;
    extern __shared__ int batch_id[];
    if (t_id == 0) {
        batch_id[0] = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.npairs_kl + QUEUE_DEPTH1 - 1) / QUEUE_DEPTH1;
    int nbatches = bounds.npairs_ij * nbatches_kl;
    while (batch_id[0] < nbatches) {
        int batch_ij = batch_id[0] / nbatches_kl;
        int batch_kl = batch_id[0] % nbatches_kl;
        int ntasks = _fill_jk_tasks_s2kl(shl_quartet_idx, envs, jk, bounds,
                                         batch_ij, batch_kl);
        if (ntasks > 0) {
            _rys_vjk_ip1_0010(envs, jk, bounds, shl_quartet_idx, ntasks);
            __syncthreads();
        }
        if (t_id == 0) {
            batch_id[0] = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_vjk_ip1_0100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + sq_id;
    double *gx = rw + 512 * nroots;
    double *gy = gx + 1024;
    double *gz = gy + 1024;
    double *rjri_cache = rw_cache + 256 * (nroots*2 + 12);
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    double s0, s1, s2;
    double Ix, Iy, Iz;

    __syncthreads();
    ShellQuartet sq = shl_quartet_idx[0];
    int ish = sq.i;
    int jsh = sq.j;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double xjxi = rj[0] - ri[0];
    double yjyi = rj[1] - ri[1];
    double zjzi = rj[2] - ri[2];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double *rjri = rjri_cache + ij*6;
        rjri[0] = xjxi;
        rjri[1] = yjyi;
        rjri[2] = zjzi;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        rjri[3] = ci[ip] * cj[jp] * Kab;
        rjri[4] = aij;
        rjri[5] = ai * 2;
    }
    double gout0x, gout0y, gout0z;
    double gout1x, gout1y, gout1z;
    double gout2x, gout2y, gout2z;
    double ai2, fx, fy, fz;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        if (ksh == lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
            gout0x = 0;
            gout0y = 0;
            gout0z = 0;
            gout1x = 0;
            gout1y = 0;
            gout1z = 0;
            gout2x = 0;
            gout2y = 0;
            gout2z = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = expk[kp];
                double al = expl[lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    double *rjri = rjri_cache + ijp*6;
                    double aij = rjri[4];
                    double ai = rjri[5] * .5;
                    double aj_aij = 1. - ai / aij;
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = rk[0] + xqc;
                    double ykl = rk[1] + yqc;
                    double zkl = rk[2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    Rpq[0] = xpq;
                    Rpq[1] = ypq;
                    Rpq[2] = zpq;
                    double cicj = rjri[3];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    double theta_rr = theta * rr;
                    __syncthreads();
                    if (omega == 0) {
                        rys_roots(1, theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[(irys*2+1)*256] *= fac;
                        }
                    } else if (omega > 0) {
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                        }
                    } else {
                        double *rw1 = rw + 512;
                        rys_roots(1, theta_rr, rw1, 256, gout_id, gout_stride);
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                            rw1[(irys*2+1)*256] *= fac;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double rt = rw[irys*512];
                        double aij = rjri[4];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        for (int n = gout_id; n < 3; n += 1) {
                            if (n == 2) {
                                gz[0] = rw[irys*512+256];
                            }
                            double *_gx = gx + n * 1024;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aj_aij;
                            double c0x = Rpa - rt_aij * Rpq[n];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[256] = s1;
                            s2 = c0x * s1 + 1 * b10 * s0;
                            _gx[512] = s2;
                            s1 = _gx[512];
                            s0 = _gx[256];
                            _gx[768] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[0];
                            _gx[512] = s1 - xjxi * s0;
                        }
                        __syncthreads();
                        ai2 = rjri[5];
                        Ix = gx[512];
                        Iy = gy[0];
                        Iz = gz[0];
                        gout0x += ai2 * gx[768] * Iy * Iz;
                        gout0y += ai2 * gy[256] * Ix * Iz;
                        gout0z += ai2 * gz[256] * Ix * Iy;
                        ai2 = rjri[5];
                        Ix = gx[0];
                        Iy = gy[512];
                        Iz = gz[0];
                        gout1x += ai2 * gx[256] * Iy * Iz;
                        gout1y += ai2 * gy[768] * Ix * Iz;
                        gout1z += ai2 * gz[256] * Ix * Iy;
                        ai2 = rjri[5];
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[512];
                        gout2x += ai2 * gx[256] * Iy * Iz;
                        gout2y += ai2 * gy[256] * Ix * Iz;
                        gout2z += ai2 * gz[768] * Ix * Iy;
                    }
                }
            }
            if (task_id < ntasks) {
                int ia = bas[ish*BAS_SLOTS+ATOM_OF] - jk.atom_offset;
                double *dm = jk.dm;
                int do_j = jk.vj != NULL;
                int do_k = jk.vk != NULL;
                double *vj_x = jk.vj + (ia*3+0)*nao*nao;
                double *vj_y = jk.vj + (ia*3+1)*nao*nao;
                double *vj_z = jk.vj + (ia*3+2)*nao*nao;
                double *vk_x = jk.vk + (ia*3+0)*nao*nao;
                double *vk_y = jk.vk + (ia*3+1)*nao*nao;
                double *vk_z = jk.vk + (ia*3+2)*nao*nao;
                if (do_j) {
                    fx = gout0x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj_x+(i0+0)*nao+(j0+0), fx);
                    atomicAdd(vj_y+(i0+0)*nao+(j0+0), fy);
                    atomicAdd(vj_z+(i0+0)*nao+(j0+0), fz);
                    fx = gout1x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout1y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout1z * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj_x+(i0+0)*nao+(j0+1), fx);
                    atomicAdd(vj_y+(i0+0)*nao+(j0+1), fy);
                    atomicAdd(vj_z+(i0+0)*nao+(j0+1), fz);
                    fx = gout2x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout2y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout2z * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj_x+(i0+0)*nao+(j0+2), fx);
                    atomicAdd(vj_y+(i0+0)*nao+(j0+2), fy);
                    atomicAdd(vj_z+(i0+0)*nao+(j0+2), fz);
                    fx = gout0x * dm[(j0+0)*nao+(i0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(i0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(i0+0)];
                    fx += gout1x * dm[(j0+1)*nao+(i0+0)];
                    fy += gout1y * dm[(j0+1)*nao+(i0+0)];
                    fz += gout1z * dm[(j0+1)*nao+(i0+0)];
                    fx += gout2x * dm[(j0+2)*nao+(i0+0)];
                    fy += gout2y * dm[(j0+2)*nao+(i0+0)];
                    fz += gout2z * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj_x+(k0+0)*nao+(l0+0), fx);
                    atomicAdd(vj_y+(k0+0)*nao+(l0+0), fy);
                    atomicAdd(vj_z+(k0+0)*nao+(l0+0), fz);
                }
                if (do_k) {
                    fx = gout0x * dm[(j0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(k0+0)];
                    fx += gout1x * dm[(j0+1)*nao+(k0+0)];
                    fy += gout1y * dm[(j0+1)*nao+(k0+0)];
                    fz += gout1z * dm[(j0+1)*nao+(k0+0)];
                    fx += gout2x * dm[(j0+2)*nao+(k0+0)];
                    fy += gout2y * dm[(j0+2)*nao+(k0+0)];
                    fz += gout2z * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(l0+0), fz);
                    fx = gout0x * dm[(i0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(l0+0), fz);
                    fx = gout1x * dm[(i0+0)*nao+(k0+0)];
                    fy = gout1y * dm[(i0+0)*nao+(k0+0)];
                    fz = gout1z * dm[(i0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(j0+1)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(j0+1)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(j0+1)*nao+(l0+0), fz);
                    fx = gout2x * dm[(i0+0)*nao+(k0+0)];
                    fy = gout2y * dm[(i0+0)*nao+(k0+0)];
                    fz = gout2z * dm[(i0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(j0+2)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(j0+2)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(j0+2)*nao+(l0+0), fz);
                    fx = gout0x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(l0+0)];
                    fx += gout1x * dm[(j0+1)*nao+(l0+0)];
                    fy += gout1y * dm[(j0+1)*nao+(l0+0)];
                    fz += gout1z * dm[(j0+1)*nao+(l0+0)];
                    fx += gout2x * dm[(j0+2)*nao+(l0+0)];
                    fy += gout2y * dm[(j0+2)*nao+(l0+0)];
                    fz += gout2z * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(k0+0), fz);
                    fx = gout0x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(k0+0), fz);
                    fx = gout1x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout1y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout1z * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+1)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(j0+1)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(j0+1)*nao+(k0+0), fz);
                    fx = gout2x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout2y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout2z * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+2)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(j0+2)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(j0+2)*nao+(k0+0), fz);
                }
            }
    }
}
__global__
void rys_vjk_ip1_0100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH1;
    extern __shared__ int batch_id[];
    if (t_id == 0) {
        batch_id[0] = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.npairs_kl + QUEUE_DEPTH1 - 1) / QUEUE_DEPTH1;
    int nbatches = bounds.npairs_ij * nbatches_kl;
    while (batch_id[0] < nbatches) {
        int batch_ij = batch_id[0] / nbatches_kl;
        int batch_kl = batch_id[0] % nbatches_kl;
        int ntasks = _fill_jk_tasks_s2kl(shl_quartet_idx, envs, jk, bounds,
                                         batch_ij, batch_kl);
        if (ntasks > 0) {
            _rys_vjk_ip1_0100(envs, jk, bounds, shl_quartet_idx, ntasks);
            __syncthreads();
        }
        if (t_id == 0) {
            batch_id[0] = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_vjk_ip1_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw_cache[];
    double *rw = rw_cache + sq_id;
    double *gx = rw + 512 * nroots;
    double *gy = gx + 768;
    double *gz = gy + 768;
    double *rjri_cache = rw_cache + 256 * (nroots*2 + 9);
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    double s0, s1, s2;
    double Ix, Iy, Iz;

    __syncthreads();
    ShellQuartet sq = shl_quartet_idx[0];
    int ish = sq.i;
    int jsh = sq.j;
    double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
    double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double xjxi = rj[0] - ri[0];
    double yjyi = rj[1] - ri[1];
    double zjzi = rj[2] - ri[2];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double *rjri = rjri_cache + ij*6;
        rjri[0] = xjxi;
        rjri[1] = yjyi;
        rjri[2] = zjzi;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        rjri[3] = ci[ip] * cj[jp] * Kab;
        rjri[4] = aij;
        rjri[5] = ai * 2;
    }
    double gout0x, gout0y, gout0z;
    double gout1x, gout1y, gout1z;
    double gout2x, gout2y, gout2z;
    double ai2, fx, fy, fz;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        double fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        if (ksh == lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
            gout0x = 0;
            gout0y = 0;
            gout0z = 0;
            gout1x = 0;
            gout1y = 0;
            gout1z = 0;
            gout2x = 0;
            gout2y = 0;
            gout2z = 0;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = expk[kp];
                double al = expl[lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double xlxk = rl[0] - rk[0];
                double ylyk = rl[1] - rk[1];
                double zlzk = rl[2] - rk[2];
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    double *rjri = rjri_cache + ijp*6;
                    double aij = rjri[4];
                    double ai = rjri[5] * .5;
                    double aj_aij = 1. - ai / aij;
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = rk[0] + xqc;
                    double ykl = rk[1] + yqc;
                    double zkl = rk[2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    Rpq[0] = xpq;
                    Rpq[1] = ypq;
                    Rpq[2] = zpq;
                    double cicj = rjri[3];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    double theta_rr = theta * rr;
                    __syncthreads();
                    if (omega == 0) {
                        rys_roots(1, theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[(irys*2+1)*256] *= fac;
                        }
                    } else if (omega > 0) {
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                        }
                    } else {
                        double *rw1 = rw + 512;
                        rys_roots(1, theta_rr, rw1, 256, gout_id, gout_stride);
                        double theta_fac = omega * omega / (omega * omega + theta);
                        rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, gout_stride);
                        __syncthreads();
                        double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                        for (int irys = gout_id; irys < 1; irys += gout_stride) {
                            rw[ irys*2   *256] *= theta_fac;
                            rw[(irys*2+1)*256] *= sqrt_theta_fac;
                            rw1[(irys*2+1)*256] *= fac;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double rt = rw[irys*512];
                        double aij = rjri[4];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        for (int n = gout_id; n < 3; n += 1) {
                            if (n == 2) {
                                gz[0] = rw[irys*512+256];
                            }
                            double *_gx = gx + n * 768;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aj_aij;
                            double c0x = Rpa - rt_aij * Rpq[n];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[256] = s1;
                            s2 = c0x * s1 + 1 * b10 * s0;
                            _gx[512] = s2;
                        }
                        __syncthreads();
                        ai2 = rjri[5];
                        Ix = gx[256];
                        Iy = gy[0];
                        Iz = gz[0];
                        gout0x += (ai2 * gx[512] - 1 * gx[0]) * Iy * Iz;
                        gout0y += ai2 * gy[256] * Ix * Iz;
                        gout0z += ai2 * gz[256] * Ix * Iy;
                        ai2 = rjri[5];
                        Ix = gx[0];
                        Iy = gy[256];
                        Iz = gz[0];
                        gout1x += ai2 * gx[256] * Iy * Iz;
                        gout1y += (ai2 * gy[512] - 1 * gy[0]) * Ix * Iz;
                        gout1z += ai2 * gz[256] * Ix * Iy;
                        ai2 = rjri[5];
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[256];
                        gout2x += ai2 * gx[256] * Iy * Iz;
                        gout2y += ai2 * gy[256] * Ix * Iz;
                        gout2z += (ai2 * gz[512] - 1 * gz[0]) * Ix * Iy;
                    }
                }
            }
            if (task_id < ntasks) {
                int ia = bas[ish*BAS_SLOTS+ATOM_OF] - jk.atom_offset;
                double *dm = jk.dm;
                int do_j = jk.vj != NULL;
                int do_k = jk.vk != NULL;
                double *vj_x = jk.vj + (ia*3+0)*nao*nao;
                double *vj_y = jk.vj + (ia*3+1)*nao*nao;
                double *vj_z = jk.vj + (ia*3+2)*nao*nao;
                double *vk_x = jk.vk + (ia*3+0)*nao*nao;
                double *vk_y = jk.vk + (ia*3+1)*nao*nao;
                double *vk_z = jk.vk + (ia*3+2)*nao*nao;
                if (do_j) {
                    fx = gout0x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj_x+(i0+0)*nao+(j0+0), fx);
                    atomicAdd(vj_y+(i0+0)*nao+(j0+0), fy);
                    atomicAdd(vj_z+(i0+0)*nao+(j0+0), fz);
                    fx = gout1x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout1y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout1z * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj_x+(i0+1)*nao+(j0+0), fx);
                    atomicAdd(vj_y+(i0+1)*nao+(j0+0), fy);
                    atomicAdd(vj_z+(i0+1)*nao+(j0+0), fz);
                    fx = gout2x * dm[(l0+0)*nao+(k0+0)];
                    fy = gout2y * dm[(l0+0)*nao+(k0+0)];
                    fz = gout2z * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj_x+(i0+2)*nao+(j0+0), fx);
                    atomicAdd(vj_y+(i0+2)*nao+(j0+0), fy);
                    atomicAdd(vj_z+(i0+2)*nao+(j0+0), fz);
                    fx = gout0x * dm[(j0+0)*nao+(i0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(i0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(i0+0)];
                    fx += gout1x * dm[(j0+0)*nao+(i0+1)];
                    fy += gout1y * dm[(j0+0)*nao+(i0+1)];
                    fz += gout1z * dm[(j0+0)*nao+(i0+1)];
                    fx += gout2x * dm[(j0+0)*nao+(i0+2)];
                    fy += gout2y * dm[(j0+0)*nao+(i0+2)];
                    fz += gout2z * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj_x+(k0+0)*nao+(l0+0), fx);
                    atomicAdd(vj_y+(k0+0)*nao+(l0+0), fy);
                    atomicAdd(vj_z+(k0+0)*nao+(l0+0), fz);
                }
                if (do_k) {
                    fx = gout0x * dm[(j0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(l0+0), fz);
                    fx = gout1x * dm[(j0+0)*nao+(k0+0)];
                    fy = gout1y * dm[(j0+0)*nao+(k0+0)];
                    fz = gout1z * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(i0+1)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(i0+1)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(i0+1)*nao+(l0+0), fz);
                    fx = gout2x * dm[(j0+0)*nao+(k0+0)];
                    fy = gout2y * dm[(j0+0)*nao+(k0+0)];
                    fz = gout2z * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk_x+(i0+2)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(i0+2)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(i0+2)*nao+(l0+0), fz);
                    fx = gout0x * dm[(i0+0)*nao+(k0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(k0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(k0+0)];
                    fx += gout1x * dm[(i0+1)*nao+(k0+0)];
                    fy += gout1y * dm[(i0+1)*nao+(k0+0)];
                    fz += gout1z * dm[(i0+1)*nao+(k0+0)];
                    fx += gout2x * dm[(i0+2)*nao+(k0+0)];
                    fy += gout2y * dm[(i0+2)*nao+(k0+0)];
                    fz += gout2z * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(l0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(l0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(l0+0), fz);
                    fx = gout0x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(i0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(i0+0)*nao+(k0+0), fz);
                    fx = gout1x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout1y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout1z * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+1)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(i0+1)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(i0+1)*nao+(k0+0), fz);
                    fx = gout2x * dm[(j0+0)*nao+(l0+0)];
                    fy = gout2y * dm[(j0+0)*nao+(l0+0)];
                    fz = gout2z * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk_x+(i0+2)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(i0+2)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(i0+2)*nao+(k0+0), fz);
                    fx = gout0x * dm[(i0+0)*nao+(l0+0)];
                    fy = gout0y * dm[(i0+0)*nao+(l0+0)];
                    fz = gout0z * dm[(i0+0)*nao+(l0+0)];
                    fx += gout1x * dm[(i0+1)*nao+(l0+0)];
                    fy += gout1y * dm[(i0+1)*nao+(l0+0)];
                    fz += gout1z * dm[(i0+1)*nao+(l0+0)];
                    fx += gout2x * dm[(i0+2)*nao+(l0+0)];
                    fy += gout2y * dm[(i0+2)*nao+(l0+0)];
                    fz += gout2z * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk_x+(j0+0)*nao+(k0+0), fx);
                    atomicAdd(vk_y+(j0+0)*nao+(k0+0), fy);
                    atomicAdd(vk_z+(j0+0)*nao+(k0+0), fz);
                }
            }
    }
}
__global__
void rys_vjk_ip1_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH1;
    extern __shared__ int batch_id[];
    if (t_id == 0) {
        batch_id[0] = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.npairs_kl + QUEUE_DEPTH1 - 1) / QUEUE_DEPTH1;
    int nbatches = bounds.npairs_ij * nbatches_kl;
    while (batch_id[0] < nbatches) {
        int batch_ij = batch_id[0] / nbatches_kl;
        int batch_kl = batch_id[0] % nbatches_kl;
        int ntasks = _fill_jk_tasks_s2kl(shl_quartet_idx, envs, jk, bounds,
                                         batch_ij, batch_kl);
        if (ntasks > 0) {
            _rys_vjk_ip1_1000(envs, jk, bounds, shl_quartet_idx, ntasks);
            __syncthreads();
        }
        if (t_id == 0) {
            batch_id[0] = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_vjk_ip1_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head, int *scheme, int workers)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int ijkl = li*125 + lj*25 + lk*5 + ll;
    int nroots = bounds->nroots;
    int g_size = bounds->stride_l * (bounds->ll + 1);
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ij_prims = iprim * jprim;
    int buflen = ij_prims*6;
    int nsq_per_block = 256;
    int gout_stride = 1;

    switch (ijkl) {
    case 0:
        break;
    case 5:
        break;
    case 25:
        break;
    case 125:
        break;
    }

#if CUDA_VERSION >= 12040
    switch (ijkl) {
    }
#endif

    dim3 threads(nsq_per_block, gout_stride);
    buflen += nroots*2 * nsq_per_block;
    switch (ijkl) {
    case 0:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_vjk_ip1_0000, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_vjk_ip1_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 5:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_vjk_ip1_0010, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_vjk_ip1_0010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 25:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_vjk_ip1_0100, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_vjk_ip1_0100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 125:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_vjk_ip1_1000, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_vjk_ip1_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    return 1;
}
