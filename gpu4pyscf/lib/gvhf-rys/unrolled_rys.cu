#include <cuda.h>
#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks.cu"


__device__ static
void _rys_jk_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4 + sq_id;
    for (int n = sq_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
        int ijp = n / TILE2;
        int sh_ij = n % TILE2;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int ip = ijp / jprim;
        int jp = ijp % jprim;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ijp * TILE2*4;
        Rpa[sh_ij+0*TILE2] = xjxi * aj_aij;
        Rpa[sh_ij+1*TILE2] = yjyi * aj_aij;
        Rpa[sh_ij+2*TILE2] = zjzi * aj_aij;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    double gout0;
    double val;
    double *dm, *vj, *vk;

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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp * TILE2*4;
                double cicj = Rpa[sh_ij+3*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = Rpa[sh_ij+0*TILE2];
                double ypa = Rpa[sh_ij+1*TILE2];
                double zpa = Rpa[sh_ij+2*TILE2];
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(1, theta_rr, rw, nsq_per_block, 0, 1);
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, nsq_per_block, 0, 1);
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 1; ++irys) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 2*nsq_per_block;
                    rys_roots(1, theta_rr, rw1, nsq_per_block, 0, 1);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, nsq_per_block, 0, 1);
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = 0; irys < 1; ++irys) {
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        gout0 += 1 * 1 * wt;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
void rys_jk_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_0000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 512;
    double *gy = gx + 512;
    double *gz = gy + 512;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 256) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 256) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(1, theta_rr, rw, 256, gout_id, 1);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 1; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 512;
                    rys_roots(1, theta_rr, rw1, 256, gout_id, 1);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 1; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*512];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    for (int n = gout_id; n < 3; n += 1) {
                        if (n == 2) {
                            gz[0] = rw[irys*512+256];
                        }
                        double *_gx = gx + n * 512;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[256] = s1;
                    }
                    __syncthreads();
                        gout0 += gx[256] * gy[0] * gz[0];
                        gout1 += gx[0] * gy[256] * gz[0];
                        gout2 += gx[0] * gy[0] * gz[256];
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 1024;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 256) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 256) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 256, gout_id, 1);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 1024;
                    rys_roots(2, theta_rr, rw1, 256, gout_id, 1);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*512];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    for (int n = gout_id; n < 3; n += 1) {
                        if (n == 2) {
                            gz[0] = rw[irys*512+256];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
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
                        gout0 += gx[768] * gy[0] * gz[0];
                        gout1 += gx[512] * gy[256] * gz[0];
                        gout2 += gx[512] * gy[0] * gz[256];
                        gout3 += gx[256] * gy[512] * gz[0];
                        gout4 += gx[0] * gy[768] * gz[0];
                        gout5 += gx[0] * gy[512] * gz[256];
                        gout6 += gx[256] * gy[0] * gz[512];
                        gout7 += gx[0] * gy[256] * gz[512];
                        gout8 += gx[0] * gy[0] * gz[768];
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout7 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout8 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+0)];
                    val += gout4 * dm[(j0+0)*nao+(i0+1)];
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+1)];
                    val += gout6 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout3 * dm[(i0+0)*nao+(k0+1)];
                    val += gout6 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout4 * dm[(i0+1)*nao+(k0+1)];
                    val += gout7 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout5 * dm[(i0+2)*nao+(k0+1)];
                    val += gout8 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 512;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 512;
                    rys_roots(2, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/akl * (1 - rt_akl);
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[128] = s1;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[256] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        _gx[512] = s2;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[384] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 1 * b00 * _gx[256];
                        _gx[640] = s2;
                        s1 = _gx[512];
                        s0 = _gx[256];
                        _gx[768] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[512] = s1 - xlxk * s0;
                        s1 = _gx[640];
                        s0 = _gx[384];
                        _gx[896] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[640] = s1 - xlxk * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[896] * gy[0] * gz[0];
                        gout1 += gx[768] * gy[0] * gz[128];
                        gout2 += gx[512] * gy[384] * gz[0];
                        gout3 += gx[640] * gy[0] * gz[256];
                        gout4 += gx[512] * gy[0] * gz[384];
                        gout5 += gx[256] * gy[640] * gz[0];
                        gout6 += gx[128] * gy[768] * gz[0];
                        gout7 += gx[0] * gy[768] * gz[128];
                        gout8 += gx[0] * gy[640] * gz[256];
                        gout9 += gx[384] * gy[0] * gz[512];
                        gout10 += gx[256] * gy[0] * gz[640];
                        gout11 += gx[0] * gy[384] * gz[512];
                        gout12 += gx[128] * gy[0] * gz[768];
                        gout13 += gx[0] * gy[0] * gz[896];
                        break;
                    case 1:
                        gout0 += gx[768] * gy[128] * gz[0];
                        gout1 += gx[640] * gy[256] * gz[0];
                        gout2 += gx[512] * gy[256] * gz[128];
                        gout3 += gx[512] * gy[128] * gz[256];
                        gout4 += gx[384] * gy[512] * gz[0];
                        gout5 += gx[256] * gy[512] * gz[128];
                        gout6 += gx[0] * gy[896] * gz[0];
                        gout7 += gx[128] * gy[512] * gz[256];
                        gout8 += gx[0] * gy[512] * gz[384];
                        gout9 += gx[256] * gy[128] * gz[512];
                        gout10 += gx[128] * gy[256] * gz[512];
                        gout11 += gx[0] * gy[256] * gz[640];
                        gout12 += gx[0] * gy[128] * gz[768];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+2)];
                    val += gout6 * dm[(l0+1)*nao+(k0+1)];
                    val += gout9 * dm[(l0+2)*nao+(k0+0)];
                    val += gout12 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout5 * dm[(l0+1)*nao+(k0+0)];
                    val += gout8 * dm[(l0+1)*nao+(k0+2)];
                    val += gout11 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+2)];
                    val += gout7 * dm[(l0+1)*nao+(k0+1)];
                    val += gout10 * dm[(l0+2)*nao+(k0+0)];
                    val += gout13 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+0)];
                    val += gout4 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+0)];
                    val += gout13 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+1)];
                    val += gout4 * dm[(l0+1)*nao+(k0+0)];
                    val += gout7 * dm[(l0+1)*nao+(k0+2)];
                    val += gout10 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+2)];
                    val += gout6 * dm[(l0+1)*nao+(k0+1)];
                    val += gout9 * dm[(l0+2)*nao+(k0+0)];
                    val += gout12 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout5 * dm[(l0+1)*nao+(k0+0)];
                    val += gout8 * dm[(l0+1)*nao+(k0+2)];
                    val += gout11 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(i0+0)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+0)];
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(i0+0)];
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(i0+0)];
                    val += gout11 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+1)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(k0+0)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout3 * dm[(i0+0)*nao+(k0+2)];
                    val += gout2 * dm[(i0+1)*nao+(k0+1)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout4 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+1)];
                    val += gout5 * dm[(i0+1)*nao+(k0+0)];
                    val += gout8 * dm[(i0+1)*nao+(k0+2)];
                    val += gout7 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout11 * dm[(i0+1)*nao+(k0+1)];
                    val += gout10 * dm[(i0+2)*nao+(k0+0)];
                    val += gout13 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout9 * dm[(i0+0)*nao+(l0+2)];
                    val += gout5 * dm[(i0+1)*nao+(l0+1)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    val += gout10 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+1)];
                    val += gout2 * dm[(i0+1)*nao+(l0+0)];
                    val += gout11 * dm[(i0+1)*nao+(l0+2)];
                    val += gout7 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout12 * dm[(i0+0)*nao+(l0+2)];
                    val += gout8 * dm[(i0+1)*nao+(l0+1)];
                    val += gout4 * dm[(i0+2)*nao+(l0+0)];
                    val += gout13 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(k0+1)];
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout3 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+0)*nao+(k0+2)];
                    val += gout6 * dm[(i0+1)*nao+(k0+1)];
                    val += gout5 * dm[(i0+2)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(k0+1)];
                    val += gout9 * dm[(i0+1)*nao+(k0+0)];
                    val += gout12 * dm[(i0+1)*nao+(k0+2)];
                    val += gout11 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(l0+1)];
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout9 * dm[(i0+1)*nao+(l0+2)];
                    val += gout5 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+0)*nao+(l0+2)];
                    val += gout6 * dm[(i0+1)*nao+(l0+1)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(l0+1)];
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout12 * dm[(i0+1)*nao+(l0+2)];
                    val += gout8 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 1024;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 256) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 256) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 256, gout_id, 1);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 1024;
                    rys_roots(2, theta_rr, rw1, 256, gout_id, 1);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*512];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    for (int n = gout_id; n < 3; n += 1) {
                        if (n == 2) {
                            gz[0] = rw[irys*512+256];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
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
                        gout0 += gx[768] * gy[0] * gz[0];
                        gout1 += gx[512] * gy[256] * gz[0];
                        gout2 += gx[512] * gy[0] * gz[256];
                        gout3 += gx[256] * gy[512] * gz[0];
                        gout4 += gx[0] * gy[768] * gz[0];
                        gout5 += gx[0] * gy[512] * gz[256];
                        gout6 += gx[256] * gy[0] * gz[512];
                        gout7 += gx[0] * gy[256] * gz[512];
                        gout8 += gx[0] * gy[0] * gz[768];
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+1)*nao+(i0+0)];
                    val += gout4 * dm[(j0+1)*nao+(i0+1)];
                    val += gout5 * dm[(j0+1)*nao+(i0+2)];
                    val += gout6 * dm[(j0+2)*nao+(i0+0)];
                    val += gout7 * dm[(j0+2)*nao+(i0+1)];
                    val += gout8 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout4 * dm[(i0+1)*nao+(k0+0)];
                    val += gout5 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 512;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 512;
                    rys_roots(2, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[128] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[256] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[512] = s1;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[640] = s1;
                        s0 = _gx[256];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[128];
                        _gx[768] = s1;
                        s1 = _gx[256];
                        s0 = _gx[128];
                        _gx[384] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[256] = s1 - xjxi * s0;
                        s1 = _gx[768];
                        s0 = _gx[640];
                        _gx[896] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[512];
                        _gx[768] = s1 - xjxi * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[896] * gy[0] * gz[0];
                        gout1 += gx[768] * gy[0] * gz[128];
                        gout2 += gx[512] * gy[384] * gz[0];
                        gout3 += gx[640] * gy[0] * gz[256];
                        gout4 += gx[512] * gy[0] * gz[384];
                        gout5 += gx[256] * gy[640] * gz[0];
                        gout6 += gx[128] * gy[768] * gz[0];
                        gout7 += gx[0] * gy[768] * gz[128];
                        gout8 += gx[0] * gy[640] * gz[256];
                        gout9 += gx[384] * gy[0] * gz[512];
                        gout10 += gx[256] * gy[0] * gz[640];
                        gout11 += gx[0] * gy[384] * gz[512];
                        gout12 += gx[128] * gy[0] * gz[768];
                        gout13 += gx[0] * gy[0] * gz[896];
                        break;
                    case 1:
                        gout0 += gx[768] * gy[128] * gz[0];
                        gout1 += gx[640] * gy[256] * gz[0];
                        gout2 += gx[512] * gy[256] * gz[128];
                        gout3 += gx[512] * gy[128] * gz[256];
                        gout4 += gx[384] * gy[512] * gz[0];
                        gout5 += gx[256] * gy[512] * gz[128];
                        gout6 += gx[0] * gy[896] * gz[0];
                        gout7 += gx[128] * gy[512] * gz[256];
                        gout8 += gx[0] * gy[512] * gz[384];
                        gout9 += gx[256] * gy[128] * gz[512];
                        gout10 += gx[128] * gy[256] * gz[512];
                        gout11 += gx[0] * gy[256] * gz[640];
                        gout12 += gx[0] * gy[128] * gz[768];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    val += gout2 * dm[(j0+1)*nao+(i0+1)];
                    val += gout3 * dm[(j0+2)*nao+(i0+0)];
                    val += gout4 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+1)];
                    val += gout6 * dm[(j0+1)*nao+(i0+0)];
                    val += gout7 * dm[(j0+1)*nao+(i0+2)];
                    val += gout8 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+2)];
                    val += gout11 * dm[(j0+1)*nao+(i0+1)];
                    val += gout12 * dm[(j0+2)*nao+(i0+0)];
                    val += gout13 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+1)*nao+(i0+0)];
                    val += gout2 * dm[(j0+1)*nao+(i0+2)];
                    val += gout3 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+0)];
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    val += gout6 * dm[(j0+1)*nao+(i0+1)];
                    val += gout7 * dm[(j0+2)*nao+(i0+0)];
                    val += gout8 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+1)];
                    val += gout10 * dm[(j0+1)*nao+(i0+0)];
                    val += gout11 * dm[(j0+1)*nao+(i0+2)];
                    val += gout12 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+2)];
                    val += gout6 * dm[(j0+1)*nao+(k0+1)];
                    val += gout3 * dm[(j0+2)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout2 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+2)];
                    val += gout7 * dm[(j0+1)*nao+(k0+1)];
                    val += gout4 * dm[(j0+2)*nao+(k0+0)];
                    val += gout13 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout9 * dm[(i0+0)*nao+(k0+2)];
                    val += gout5 * dm[(i0+1)*nao+(k0+1)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout10 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+1)];
                    val += gout2 * dm[(i0+1)*nao+(k0+0)];
                    val += gout11 * dm[(i0+1)*nao+(k0+2)];
                    val += gout7 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout8 * dm[(i0+1)*nao+(k0+1)];
                    val += gout4 * dm[(i0+2)*nao+(k0+0)];
                    val += gout13 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout1 * dm[(j0+1)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+2)];
                    val += gout7 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+2)];
                    val += gout6 * dm[(j0+1)*nao+(k0+1)];
                    val += gout3 * dm[(j0+2)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout2 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(k0+1)];
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout9 * dm[(i0+1)*nao+(k0+2)];
                    val += gout5 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(k0+0)];
                    val += gout10 * dm[(i0+0)*nao+(k0+2)];
                    val += gout6 * dm[(i0+1)*nao+(k0+1)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout11 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(k0+1)];
                    val += gout3 * dm[(i0+1)*nao+(k0+0)];
                    val += gout12 * dm[(i0+1)*nao+(k0+2)];
                    val += gout8 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(l0+0)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+1)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_1111(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 384;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 64) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double gout18;
    double gout19;
    double gout20;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 64) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
        gout18 = 0;
        gout19 = 0;
        gout20 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, 64, gout_id, 4);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 4) {
                        rw[ irys*2   *64] *= theta_fac;
                        rw[(irys*2+1)*64] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 384;
                    rys_roots(3, theta_rr, rw1, 64, gout_id, 4);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 4) {
                        rw[ irys*2   *64] *= theta_fac;
                        rw[(irys*2+1)*64] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/akl * (1 - rt_akl);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gz[0] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
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
                        s1 = _gx[512];
                        s0 = _gx[256];
                        _gx[768] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[512] = s1 - xlxk * s0;
                        s1 = _gx[576];
                        s0 = _gx[320];
                        _gx[832] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[576] = s1 - xlxk * s0;
                        s1 = _gx[640];
                        s0 = _gx[384];
                        _gx[896] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[640] = s1 - xlxk * s0;
                        s1 = _gx[704];
                        s0 = _gx[448];
                        _gx[960] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[192];
                        _gx[704] = s1 - xlxk * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[960] * gy[0] * gz[0];
                        gout1 += gx[768] * gy[192] * gz[0];
                        gout2 += gx[768] * gy[0] * gz[192];
                        gout3 += gx[576] * gy[384] * gz[0];
                        gout4 += gx[512] * gy[320] * gz[128];
                        gout5 += gx[640] * gy[0] * gz[320];
                        gout6 += gx[576] * gy[0] * gz[384];
                        gout7 += gx[384] * gy[576] * gz[0];
                        gout8 += gx[256] * gy[640] * gz[64];
                        gout9 += gx[192] * gy[768] * gz[0];
                        gout10 += gx[0] * gy[960] * gz[0];
                        gout11 += gx[0] * gy[768] * gz[192];
                        gout12 += gx[64] * gy[640] * gz[256];
                        gout13 += gx[0] * gy[576] * gz[384];
                        gout14 += gx[384] * gy[0] * gz[576];
                        gout15 += gx[320] * gy[0] * gz[640];
                        gout16 += gx[128] * gy[320] * gz[512];
                        gout17 += gx[0] * gy[384] * gz[576];
                        gout18 += gx[192] * gy[0] * gz[768];
                        gout19 += gx[0] * gy[192] * gz[768];
                        gout20 += gx[0] * gy[0] * gz[960];
                        break;
                    case 1:
                        gout0 += gx[896] * gy[64] * gz[0];
                        gout1 += gx[768] * gy[128] * gz[64];
                        gout2 += gx[704] * gy[256] * gz[0];
                        gout3 += gx[512] * gy[448] * gz[0];
                        gout4 += gx[512] * gy[256] * gz[192];
                        gout5 += gx[576] * gy[128] * gz[256];
                        gout6 += gx[512] * gy[64] * gz[384];
                        gout7 += gx[384] * gy[512] * gz[64];
                        gout8 += gx[320] * gy[512] * gz[128];
                        gout9 += gx[128] * gy[832] * gz[0];
                        gout10 += gx[0] * gy[896] * gz[64];
                        gout11 += gx[192] * gy[512] * gz[256];
                        gout12 += gx[0] * gy[704] * gz[256];
                        gout13 += gx[0] * gy[512] * gz[448];
                        gout14 += gx[320] * gy[128] * gz[512];
                        gout15 += gx[256] * gy[64] * gz[640];
                        gout16 += gx[128] * gy[256] * gz[576];
                        gout17 += gx[64] * gy[256] * gz[640];
                        gout18 += gx[128] * gy[64] * gz[768];
                        gout19 += gx[0] * gy[128] * gz[832];
                        break;
                    case 2:
                        gout0 += gx[896] * gy[0] * gz[64];
                        gout1 += gx[832] * gy[0] * gz[128];
                        gout2 += gx[640] * gy[320] * gz[0];
                        gout3 += gx[512] * gy[384] * gz[64];
                        gout4 += gx[704] * gy[0] * gz[256];
                        gout5 += gx[512] * gy[192] * gz[256];
                        gout6 += gx[512] * gy[0] * gz[448];
                        gout7 += gx[320] * gy[640] * gz[0];
                        gout8 += gx[256] * gy[576] * gz[128];
                        gout9 += gx[128] * gy[768] * gz[64];
                        gout10 += gx[64] * gy[768] * gz[128];
                        gout11 += gx[128] * gy[576] * gz[256];
                        gout12 += gx[0] * gy[640] * gz[320];
                        gout13 += gx[448] * gy[0] * gz[512];
                        gout14 += gx[256] * gy[192] * gz[512];
                        gout15 += gx[256] * gy[0] * gz[704];
                        gout16 += gx[64] * gy[384] * gz[512];
                        gout17 += gx[0] * gy[320] * gz[640];
                        gout18 += gx[128] * gy[0] * gz[832];
                        gout19 += gx[64] * gy[0] * gz[896];
                        break;
                    case 3:
                        gout0 += gx[832] * gy[128] * gz[0];
                        gout1 += gx[768] * gy[64] * gz[128];
                        gout2 += gx[640] * gy[256] * gz[64];
                        gout3 += gx[576] * gy[256] * gz[128];
                        gout4 += gx[640] * gy[64] * gz[256];
                        gout5 += gx[512] * gy[128] * gz[320];
                        gout6 += gx[448] * gy[512] * gz[0];
                        gout7 += gx[256] * gy[704] * gz[0];
                        gout8 += gx[256] * gy[512] * gz[192];
                        gout9 += gx[64] * gy[896] * gz[0];
                        gout10 += gx[0] * gy[832] * gz[128];
                        gout11 += gx[128] * gy[512] * gz[320];
                        gout12 += gx[64] * gy[512] * gz[384];
                        gout13 += gx[384] * gy[64] * gz[512];
                        gout14 += gx[256] * gy[128] * gz[576];
                        gout15 += gx[192] * gy[256] * gz[512];
                        gout16 += gx[0] * gy[448] * gz[512];
                        gout17 += gx[0] * gy[256] * gz[704];
                        gout18 += gx[64] * gy[128] * gz[768];
                        gout19 += gx[0] * gy[64] * gz[896];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+1)*nao+(k0+1)];
                    val += gout18 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    val += gout15 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout7 * dm[(l0+1)*nao+(k0+0)];
                    val += gout16 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+1)*nao+(k0+1)];
                    val += gout19 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout13 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+2)];
                    val += gout14 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+1)*nao+(k0+0)];
                    val += gout17 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+1)*nao+(k0+1)];
                    val += gout20 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+1)*nao+(i0+1)];
                    val += gout2 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+1)*nao+(i0+0)];
                    val += gout4 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    val += gout6 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+1)*nao+(i0+1)];
                    val += gout11 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout12 * dm[(j0+1)*nao+(i0+0)];
                    val += gout13 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(i0+2)];
                    val += gout15 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(i0+1)];
                    val += gout17 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+0)];
                    val += gout19 * dm[(j0+1)*nao+(i0+1)];
                    val += gout20 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout11 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+2)];
                    val += gout14 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+1)*nao+(k0+0)];
                    val += gout17 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+1)*nao+(k0+1)];
                    val += gout18 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    val += gout15 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout7 * dm[(l0+1)*nao+(k0+0)];
                    val += gout16 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+1)*nao+(k0+1)];
                    val += gout19 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout13 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(i0+0)];
                    val += gout3 * dm[(j0+1)*nao+(i0+1)];
                    val += gout4 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+1)*nao+(i0+0)];
                    val += gout6 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(i0+2)];
                    val += gout8 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+1)];
                    val += gout10 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(i0+0)];
                    val += gout12 * dm[(j0+1)*nao+(i0+1)];
                    val += gout13 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+1)*nao+(i0+0)];
                    val += gout15 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(i0+2)];
                    val += gout17 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+1)];
                    val += gout19 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    case 2:
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+2)];
                    val += gout13 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+1)*nao+(k0+0)];
                    val += gout16 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+1)*nao+(k0+1)];
                    val += gout19 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout11 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+2)];
                    val += gout14 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+1)*nao+(k0+0)];
                    val += gout17 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+1)*nao+(k0+1)];
                    val += gout18 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    val += gout15 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+2)];
                    val += gout1 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(i0+1)];
                    val += gout3 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+0)];
                    val += gout5 * dm[(j0+1)*nao+(i0+1)];
                    val += gout6 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+1)*nao+(i0+0)];
                    val += gout8 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+2)];
                    val += gout10 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(i0+1)];
                    val += gout12 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(i0+0)];
                    val += gout14 * dm[(j0+1)*nao+(i0+1)];
                    val += gout15 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout16 * dm[(j0+1)*nao+(i0+0)];
                    val += gout17 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(i0+2)];
                    val += gout19 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    case 3:
                    val = 0;
                    val += gout6 * dm[(l0+1)*nao+(k0+0)];
                    val += gout15 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+1)*nao+(k0+1)];
                    val += gout18 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+2)];
                    val += gout13 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+1)*nao+(k0+0)];
                    val += gout16 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+1)*nao+(k0+1)];
                    val += gout19 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout11 * dm[(l0+1)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+2)];
                    val += gout14 * dm[(l0+2)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+1)*nao+(k0+0)];
                    val += gout17 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+1)*nao+(i0+0)];
                    val += gout1 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+1)];
                    val += gout5 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+1)*nao+(i0+1)];
                    val += gout8 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+1)*nao+(i0+0)];
                    val += gout10 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(i0+2)];
                    val += gout12 * dm[(j0+2)*nao+(i0+0)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(i0+1)];
                    val += gout14 * dm[(j0+1)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(i0+0)];
                    val += gout16 * dm[(j0+1)*nao+(i0+1)];
                    val += gout17 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout18 * dm[(j0+1)*nao+(i0+0)];
                    val += gout19 * dm[(j0+2)*nao+(i0+1)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+1)];
                    val += gout6 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(k0+2)];
                    val += gout15 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+1)*nao+(k0+0)];
                    val += gout4 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+1)];
                    val += gout13 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(k0+1)];
                    val += gout19 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+2)];
                    val += gout2 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(k0+0)];
                    val += gout17 * dm[(j0+1)*nao+(k0+1)];
                    val += gout20 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout5 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(k0+1)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(k0+2)];
                    val += gout16 * dm[(i0+1)*nao+(k0+1)];
                    val += gout14 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+1)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout10 * dm[(i0+1)*nao+(k0+1)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout19 * dm[(i0+1)*nao+(k0+2)];
                    val += gout17 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+2)];
                    val += gout4 * dm[(i0+1)*nao+(k0+1)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout13 * dm[(i0+1)*nao+(k0+2)];
                    val += gout11 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(k0+0)];
                    val += gout20 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout15 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+1)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+2)];
                    val += gout12 * dm[(j0+1)*nao+(l0+1)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    val += gout1 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+2)];
                    val += gout10 * dm[(j0+1)*nao+(l0+1)];
                    val += gout4 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout19 * dm[(j0+1)*nao+(l0+2)];
                    val += gout13 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+2)];
                    val += gout8 * dm[(j0+1)*nao+(l0+1)];
                    val += gout2 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout17 * dm[(j0+1)*nao+(l0+2)];
                    val += gout11 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout20 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+1)];
                    val += gout14 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+1)];
                    val += gout16 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(l0+2)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+1)*nao+(l0+1)];
                    val += gout17 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+1)];
                    val += gout19 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(l0+2)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+1)*nao+(l0+1)];
                    val += gout20 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    val += gout5 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+1)*nao+(k0+0)];
                    val += gout17 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+1)];
                    val += gout6 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(k0+2)];
                    val += gout15 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+1)*nao+(k0+0)];
                    val += gout4 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+1)];
                    val += gout13 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(k0+1)];
                    val += gout19 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(i0+0)*nao+(k0+1)];
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout11 * dm[(i0+0)*nao+(k0+2)];
                    val += gout9 * dm[(i0+1)*nao+(k0+1)];
                    val += gout7 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+1)*nao+(k0+2)];
                    val += gout16 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout5 * dm[(i0+0)*nao+(k0+2)];
                    val += gout3 * dm[(i0+1)*nao+(k0+1)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+1)*nao+(k0+2)];
                    val += gout10 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout14 * dm[(i0+0)*nao+(k0+0)];
                    val += gout19 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(k0+2)];
                    val += gout4 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout8 * dm[(i0+0)*nao+(k0+0)];
                    val += gout13 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout17 * dm[(i0+0)*nao+(k0+1)];
                    val += gout15 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout14 * dm[(j0+1)*nao+(l0+2)];
                    val += gout8 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout17 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+1)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout15 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+1)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+2)];
                    val += gout12 * dm[(j0+1)*nao+(l0+1)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    val += gout1 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+2)];
                    val += gout10 * dm[(j0+1)*nao+(l0+1)];
                    val += gout4 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout19 * dm[(j0+1)*nao+(l0+2)];
                    val += gout13 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout7 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(i0+0)*nao+(l0+0)];
                    val += gout9 * dm[(i0+1)*nao+(l0+1)];
                    val += gout16 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(i0+0)*nao+(l0+1)];
                    val += gout18 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout14 * dm[(i0+0)*nao+(l0+2)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout10 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout5 * dm[(i0+0)*nao+(l0+0)];
                    val += gout12 * dm[(i0+1)*nao+(l0+1)];
                    val += gout19 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout8 * dm[(i0+0)*nao+(l0+1)];
                    val += gout15 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout17 * dm[(i0+0)*nao+(l0+2)];
                    val += gout4 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+0)];
                    val += gout13 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    case 2:
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+2)];
                    val += gout1 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+1)*nao+(k0+0)];
                    val += gout10 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(k0+0)];
                    val += gout16 * dm[(j0+1)*nao+(k0+1)];
                    val += gout19 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    val += gout5 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+1)*nao+(k0+0)];
                    val += gout17 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+1)];
                    val += gout6 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(k0+2)];
                    val += gout15 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(k0+2)];
                    val += gout2 * dm[(i0+1)*nao+(k0+1)];
                    val += gout0 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout11 * dm[(i0+1)*nao+(k0+2)];
                    val += gout9 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout13 * dm[(i0+0)*nao+(k0+0)];
                    val += gout18 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout5 * dm[(i0+1)*nao+(k0+2)];
                    val += gout3 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout16 * dm[(i0+0)*nao+(k0+1)];
                    val += gout14 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(k0+0)];
                    val += gout6 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(k0+1)];
                    val += gout8 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout19 * dm[(i0+0)*nao+(k0+2)];
                    val += gout17 * dm[(i0+1)*nao+(k0+1)];
                    val += gout15 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+2)];
                    val += gout7 * dm[(j0+1)*nao+(l0+1)];
                    val += gout1 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout16 * dm[(j0+1)*nao+(l0+2)];
                    val += gout10 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout19 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout14 * dm[(j0+1)*nao+(l0+2)];
                    val += gout8 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout17 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+1)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout15 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+1)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+0)*nao+(l0+2)];
                    val += gout12 * dm[(j0+1)*nao+(l0+1)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout13 * dm[(i0+0)*nao+(l0+2)];
                    val += gout0 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(i0+1)*nao+(l0+0)];
                    val += gout9 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(l0+0)];
                    val += gout11 * dm[(i0+1)*nao+(l0+1)];
                    val += gout18 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(l0+1)];
                    val += gout14 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout16 * dm[(i0+0)*nao+(l0+2)];
                    val += gout3 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout5 * dm[(i0+1)*nao+(l0+0)];
                    val += gout12 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(l0+0)];
                    val += gout8 * dm[(i0+1)*nao+(l0+1)];
                    val += gout15 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(l0+1)];
                    val += gout17 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout19 * dm[(i0+0)*nao+(l0+2)];
                    val += gout6 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    case 3:
                    val = 0;
                    val += gout0 * dm[(j0+1)*nao+(k0+0)];
                    val += gout3 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+1)*nao+(k0+1)];
                    val += gout12 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(k0+1)];
                    val += gout18 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+2)];
                    val += gout1 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+1)*nao+(k0+0)];
                    val += gout10 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(k0+0)];
                    val += gout16 * dm[(j0+1)*nao+(k0+1)];
                    val += gout19 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    val += gout5 * dm[(j0+1)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+1)*nao+(k0+0)];
                    val += gout17 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout4 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout11 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(k0+1)];
                    val += gout13 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout5 * dm[(i0+2)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(k0+1)];
                    val += gout7 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(k0+2)];
                    val += gout16 * dm[(i0+1)*nao+(k0+1)];
                    val += gout14 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+1)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout10 * dm[(i0+1)*nao+(k0+1)];
                    val += gout8 * dm[(i0+2)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout19 * dm[(i0+1)*nao+(k0+2)];
                    val += gout17 * dm[(i0+2)*nao+(k0+1)];
                    atomicAdd(vk+(j0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+1)];
                    val += gout0 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(l0+2)];
                    val += gout9 * dm[(j0+1)*nao+(l0+1)];
                    val += gout3 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(j0+1)*nao+(l0+2)];
                    val += gout12 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+2)];
                    val += gout7 * dm[(j0+1)*nao+(l0+1)];
                    val += gout1 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout16 * dm[(j0+1)*nao+(l0+2)];
                    val += gout10 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout19 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout14 * dm[(j0+1)*nao+(l0+2)];
                    val += gout8 * dm[(j0+2)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout17 * dm[(j0+2)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+1)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+1)];
                    val += gout13 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(l0+2)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+1)*nao+(l0+1)];
                    val += gout14 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+1)];
                    val += gout16 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout18 * dm[(i0+0)*nao+(l0+2)];
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+2)*nao+(l0+1)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+1)*nao+(l0+1)];
                    val += gout17 * dm[(i0+2)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+1)];
                    val += gout19 * dm[(i0+1)*nao+(l0+2)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_1111(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_1111(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 1024;
    double *gy = gx + 768;
    double *gz = gy + 768;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 256) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 256) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 256, gout_id, 1);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 1024;
                    rys_roots(2, theta_rr, rw1, 256, gout_id, 1);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*512];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    for (int n = gout_id; n < 3; n += 1) {
                        if (n == 2) {
                            gz[0] = rw[irys*512+256];
                        }
                        double *_gx = gx + n * 768;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[256] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[512] = s2;
                    }
                    __syncthreads();
                        gout0 += gx[512] * gy[0] * gz[0];
                        gout1 += gx[256] * gy[256] * gz[0];
                        gout2 += gx[256] * gy[0] * gz[256];
                        gout3 += gx[0] * gy[512] * gz[0];
                        gout4 += gx[0] * gy[256] * gz[256];
                        gout5 += gx[0] * gy[0] * gz[512];
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 512;
    double *gy = gx + 768;
    double *gz = gy + 768;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 512;
                    rys_roots(2, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 768;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[128] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[256] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[384] = s1;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[512] = s1;
                        s0 = _gx[256];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[128];
                        _gx[640] = s1;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[640] * gy[0] * gz[0];
                        gout1 += gx[512] * gy[0] * gz[128];
                        gout2 += gx[384] * gy[128] * gz[128];
                        gout3 += gx[256] * gy[384] * gz[0];
                        gout4 += gx[128] * gy[384] * gz[128];
                        gout5 += gx[0] * gy[512] * gz[128];
                        gout6 += gx[256] * gy[0] * gz[384];
                        gout7 += gx[128] * gy[0] * gz[512];
                        gout8 += gx[0] * gy[128] * gz[512];
                        break;
                    case 1:
                        gout0 += gx[512] * gy[128] * gz[0];
                        gout1 += gx[384] * gy[256] * gz[0];
                        gout2 += gx[384] * gy[0] * gz[256];
                        gout3 += gx[128] * gy[512] * gz[0];
                        gout4 += gx[0] * gy[640] * gz[0];
                        gout5 += gx[0] * gy[384] * gz[256];
                        gout6 += gx[128] * gy[128] * gz[384];
                        gout7 += gx[0] * gy[256] * gz[384];
                        gout8 += gx[0] * gy[0] * gz[640];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout7 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout8 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    val += gout2 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+0)];
                    val += gout4 * dm[(j0+0)*nao+(i0+2)];
                    val += gout5 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+2)];
                    val += gout8 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout7 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout8 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+3)];
                    val += gout2 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+1)];
                    val += gout4 * dm[(j0+0)*nao+(i0+3)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+1)];
                    val += gout7 * dm[(j0+0)*nao+(i0+3)];
                    val += gout8 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+1)];
                    val += gout6 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout3 * dm[(i0+0)*nao+(k0+1)];
                    val += gout6 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout4 * dm[(i0+2)*nao+(k0+1)];
                    val += gout7 * dm[(i0+2)*nao+(k0+2)];
                    val += gout2 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+4)*nao+(k0+1)];
                    val += gout8 * dm[(i0+4)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+2)*nao+(l0+0)];
                    val += gout5 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+2)*nao+(l0+0)];
                    val += gout8 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+1)];
                    val += gout6 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout3 * dm[(i0+1)*nao+(k0+1)];
                    val += gout6 * dm[(i0+1)*nao+(k0+2)];
                    val += gout1 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+3)*nao+(k0+1)];
                    val += gout7 * dm[(i0+3)*nao+(k0+2)];
                    val += gout2 * dm[(i0+5)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+1)];
                    val += gout8 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout1 * dm[(i0+3)*nao+(l0+0)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout4 * dm[(i0+3)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+0)];
                    val += gout7 * dm[(i0+3)*nao+(l0+0)];
                    val += gout8 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 384;
    double *gy = gx + 768;
    double *gz = gy + 768;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 64) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 64) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, 64, gout_id, 4);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 4) {
                        rw[ irys*2   *64] *= theta_fac;
                        rw[(irys*2+1)*64] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 384;
                    rys_roots(3, theta_rr, rw1, 64, gout_id, 4);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 4) {
                        rw[ irys*2   *64] *= theta_fac;
                        rw[(irys*2+1)*64] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/akl * (1 - rt_akl);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gz[0] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 768;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[192] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        _gx[384] = s2;
                        s0 = _gx[64];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[256] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 1 * b00 * _gx[192];
                        _gx[448] = s2;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[64];
                        _gx[320] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 2 * b00 * _gx[256];
                        _gx[512] = s2;
                        s1 = _gx[384];
                        s0 = _gx[192];
                        _gx[576] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[384] = s1 - xlxk * s0;
                        s1 = _gx[448];
                        s0 = _gx[256];
                        _gx[640] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[448] = s1 - xlxk * s0;
                        s1 = _gx[512];
                        s0 = _gx[320];
                        _gx[704] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[512] = s1 - xlxk * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[704] * gy[0] * gz[0];
                        gout1 += gx[576] * gy[64] * gz[64];
                        gout2 += gx[448] * gy[192] * gz[64];
                        gout3 += gx[512] * gy[0] * gz[192];
                        gout4 += gx[384] * gy[64] * gz[256];
                        gout5 += gx[256] * gy[384] * gz[64];
                        gout6 += gx[128] * gy[576] * gz[0];
                        gout7 += gx[0] * gy[640] * gz[64];
                        gout8 += gx[64] * gy[384] * gz[256];
                        gout9 += gx[320] * gy[0] * gz[384];
                        gout10 += gx[192] * gy[64] * gz[448];
                        gout11 += gx[64] * gy[192] * gz[448];
                        gout12 += gx[128] * gy[0] * gz[576];
                        gout13 += gx[0] * gy[64] * gz[640];
                        break;
                    case 1:
                        gout0 += gx[640] * gy[64] * gz[0];
                        gout1 += gx[576] * gy[0] * gz[128];
                        gout2 += gx[384] * gy[320] * gz[0];
                        gout3 += gx[448] * gy[64] * gz[192];
                        gout4 += gx[384] * gy[0] * gz[320];
                        gout5 += gx[192] * gy[512] * gz[0];
                        gout6 += gx[64] * gy[640] * gz[0];
                        gout7 += gx[0] * gy[576] * gz[128];
                        gout8 += gx[0] * gy[512] * gz[192];
                        gout9 += gx[256] * gy[64] * gz[384];
                        gout10 += gx[192] * gy[0] * gz[512];
                        gout11 += gx[0] * gy[320] * gz[384];
                        gout12 += gx[64] * gy[64] * gz[576];
                        gout13 += gx[0] * gy[0] * gz[704];
                        break;
                    case 2:
                        gout0 += gx[640] * gy[0] * gz[64];
                        gout1 += gx[512] * gy[192] * gz[0];
                        gout2 += gx[384] * gy[256] * gz[64];
                        gout3 += gx[448] * gy[0] * gz[256];
                        gout4 += gx[320] * gy[384] * gz[0];
                        gout5 += gx[192] * gy[448] * gz[64];
                        gout6 += gx[64] * gy[576] * gz[64];
                        gout7 += gx[128] * gy[384] * gz[192];
                        gout8 += gx[0] * gy[448] * gz[256];
                        gout9 += gx[256] * gy[0] * gz[448];
                        gout10 += gx[128] * gy[192] * gz[384];
                        gout11 += gx[0] * gy[256] * gz[448];
                        gout12 += gx[64] * gy[0] * gz[640];
                        break;
                    case 3:
                        gout0 += gx[576] * gy[128] * gz[0];
                        gout1 += gx[448] * gy[256] * gz[0];
                        gout2 += gx[384] * gy[192] * gz[128];
                        gout3 += gx[384] * gy[128] * gz[192];
                        gout4 += gx[256] * gy[448] * gz[0];
                        gout5 += gx[192] * gy[384] * gz[128];
                        gout6 += gx[0] * gy[704] * gz[0];
                        gout7 += gx[64] * gy[448] * gz[192];
                        gout8 += gx[0] * gy[384] * gz[320];
                        gout9 += gx[192] * gy[128] * gz[384];
                        gout10 += gx[64] * gy[256] * gz[384];
                        gout11 += gx[0] * gy[192] * gz[512];
                        gout12 += gx[0] * gy[128] * gz[576];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+2)];
                    val += gout6 * dm[(l0+1)*nao+(k0+1)];
                    val += gout9 * dm[(l0+2)*nao+(k0+0)];
                    val += gout12 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout5 * dm[(l0+1)*nao+(k0+0)];
                    val += gout8 * dm[(l0+1)*nao+(k0+2)];
                    val += gout11 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+2)];
                    val += gout7 * dm[(l0+1)*nao+(k0+1)];
                    val += gout10 * dm[(l0+2)*nao+(k0+0)];
                    val += gout13 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+0)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+0)];
                    val += gout13 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+2)];
                    val += gout6 * dm[(l0+1)*nao+(k0+1)];
                    val += gout9 * dm[(l0+2)*nao+(k0+0)];
                    val += gout12 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout5 * dm[(l0+1)*nao+(k0+0)];
                    val += gout8 * dm[(l0+1)*nao+(k0+2)];
                    val += gout11 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+2)];
                    val += gout7 * dm[(l0+1)*nao+(k0+1)];
                    val += gout10 * dm[(l0+2)*nao+(k0+0)];
                    val += gout13 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+1)];
                    val += gout4 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+1)];
                    val += gout7 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+1)];
                    val += gout10 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+1)];
                    val += gout13 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    case 2:
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+1)];
                    val += gout4 * dm[(l0+1)*nao+(k0+0)];
                    val += gout7 * dm[(l0+1)*nao+(k0+2)];
                    val += gout10 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+2)];
                    val += gout6 * dm[(l0+1)*nao+(k0+1)];
                    val += gout9 * dm[(l0+2)*nao+(k0+0)];
                    val += gout12 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout5 * dm[(l0+1)*nao+(k0+0)];
                    val += gout8 * dm[(l0+1)*nao+(k0+2)];
                    val += gout11 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(i0+0)];
                    val += gout2 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+0)];
                    val += gout5 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(i0+0)];
                    val += gout8 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(i0+0)];
                    val += gout11 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    case 3:
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+1)];
                    val += gout4 * dm[(l0+1)*nao+(k0+0)];
                    val += gout7 * dm[(l0+1)*nao+(k0+2)];
                    val += gout10 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+2)];
                    val += gout6 * dm[(l0+1)*nao+(k0+1)];
                    val += gout9 * dm[(l0+2)*nao+(k0+0)];
                    val += gout12 * dm[(l0+2)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+1)];
                    val += gout5 * dm[(l0+1)*nao+(k0+0)];
                    val += gout8 * dm[(l0+1)*nao+(k0+2)];
                    val += gout11 * dm[(l0+2)*nao+(k0+1)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+1)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(i0+1)];
                    val += gout8 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(i0+1)];
                    val += gout11 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+3)];
                    atomicAdd(vj+(k0+2)*nao+(l0+2), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+4)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(k0+0)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout3 * dm[(i0+0)*nao+(k0+2)];
                    val += gout2 * dm[(i0+2)*nao+(k0+1)];
                    val += gout1 * dm[(i0+4)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+1)];
                    val += gout5 * dm[(i0+2)*nao+(k0+0)];
                    val += gout8 * dm[(i0+2)*nao+(k0+2)];
                    val += gout7 * dm[(i0+4)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout11 * dm[(i0+2)*nao+(k0+1)];
                    val += gout10 * dm[(i0+4)*nao+(k0+0)];
                    val += gout13 * dm[(i0+4)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout9 * dm[(i0+0)*nao+(l0+2)];
                    val += gout5 * dm[(i0+2)*nao+(l0+1)];
                    val += gout1 * dm[(i0+4)*nao+(l0+0)];
                    val += gout10 * dm[(i0+4)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+1)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+2)];
                    val += gout7 * dm[(i0+4)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout12 * dm[(i0+0)*nao+(l0+2)];
                    val += gout8 * dm[(i0+2)*nao+(l0+1)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout13 * dm[(i0+4)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+3)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+5)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(k0+0)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout3 * dm[(i0+1)*nao+(k0+2)];
                    val += gout2 * dm[(i0+3)*nao+(k0+1)];
                    val += gout1 * dm[(i0+5)*nao+(k0+0)];
                    val += gout4 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(k0+1)];
                    val += gout5 * dm[(i0+3)*nao+(k0+0)];
                    val += gout8 * dm[(i0+3)*nao+(k0+2)];
                    val += gout7 * dm[(i0+5)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+1)*nao+(k0+0)];
                    val += gout12 * dm[(i0+1)*nao+(k0+2)];
                    val += gout11 * dm[(i0+3)*nao+(k0+1)];
                    val += gout10 * dm[(i0+5)*nao+(k0+0)];
                    val += gout13 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout9 * dm[(i0+1)*nao+(l0+2)];
                    val += gout5 * dm[(i0+3)*nao+(l0+1)];
                    val += gout1 * dm[(i0+5)*nao+(l0+0)];
                    val += gout10 * dm[(i0+5)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+1)];
                    val += gout2 * dm[(i0+3)*nao+(l0+0)];
                    val += gout11 * dm[(i0+3)*nao+(l0+2)];
                    val += gout7 * dm[(i0+5)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout12 * dm[(i0+1)*nao+(l0+2)];
                    val += gout8 * dm[(i0+3)*nao+(l0+1)];
                    val += gout4 * dm[(i0+5)*nao+(l0+0)];
                    val += gout13 * dm[(i0+5)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    case 2:
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+4)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(k0+1)];
                    val += gout0 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+2)*nao+(k0+2)];
                    val += gout2 * dm[(i0+4)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+0)*nao+(k0+2)];
                    val += gout6 * dm[(i0+2)*nao+(k0+1)];
                    val += gout5 * dm[(i0+4)*nao+(k0+0)];
                    val += gout8 * dm[(i0+4)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(k0+1)];
                    val += gout9 * dm[(i0+2)*nao+(k0+0)];
                    val += gout12 * dm[(i0+2)*nao+(k0+2)];
                    val += gout11 * dm[(i0+4)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(l0+1)];
                    val += gout0 * dm[(i0+2)*nao+(l0+0)];
                    val += gout9 * dm[(i0+2)*nao+(l0+2)];
                    val += gout5 * dm[(i0+4)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+0)*nao+(l0+2)];
                    val += gout6 * dm[(i0+2)*nao+(l0+1)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    val += gout11 * dm[(i0+4)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(l0+1)];
                    val += gout3 * dm[(i0+2)*nao+(l0+0)];
                    val += gout12 * dm[(i0+2)*nao+(l0+2)];
                    val += gout8 * dm[(i0+4)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    case 3:
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+3)*nao+(l0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(k0+1)];
                    atomicAdd(vk+(i0+5)*nao+(l0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+1)*nao+(k0+1)];
                    val += gout0 * dm[(i0+3)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+2)];
                    val += gout2 * dm[(i0+5)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+1)*nao+(k0+0)];
                    val += gout7 * dm[(i0+1)*nao+(k0+2)];
                    val += gout6 * dm[(i0+3)*nao+(k0+1)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout8 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+1)*nao+(k0+1)];
                    val += gout9 * dm[(i0+3)*nao+(k0+0)];
                    val += gout12 * dm[(i0+3)*nao+(k0+2)];
                    val += gout11 * dm[(i0+5)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout10 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout11 * dm[(j0+0)*nao+(l0+2)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+1)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(i0+1)*nao+(l0+1)];
                    val += gout0 * dm[(i0+3)*nao+(l0+0)];
                    val += gout9 * dm[(i0+3)*nao+(l0+2)];
                    val += gout5 * dm[(i0+5)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout10 * dm[(i0+1)*nao+(l0+2)];
                    val += gout6 * dm[(i0+3)*nao+(l0+1)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+2)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(i0+1)*nao+(l0+1)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout12 * dm[(i0+3)*nao+(l0+2)];
                    val += gout8 * dm[(i0+5)*nao+(l0+1)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2011(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2020(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 768;
    double *gy = gx + 1152;
    double *gz = gy + 1152;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 768;
                    rys_roots(3, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b01 = .5/akl * (1 - rt_akl);
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 1152;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[128] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[256] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
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
                        gout0 += gx[1024] * gy[0] * gz[0];
                        gout1 += gx[896] * gy[0] * gz[128];
                        gout2 += gx[768] * gy[128] * gz[128];
                        gout3 += gx[640] * gy[384] * gz[0];
                        gout4 += gx[512] * gy[384] * gz[128];
                        gout5 += gx[384] * gy[512] * gz[128];
                        gout6 += gx[640] * gy[0] * gz[384];
                        gout7 += gx[512] * gy[0] * gz[512];
                        gout8 += gx[384] * gy[128] * gz[512];
                        gout9 += gx[256] * gy[768] * gz[0];
                        gout10 += gx[128] * gy[768] * gz[128];
                        gout11 += gx[0] * gy[896] * gz[128];
                        gout12 += gx[256] * gy[384] * gz[384];
                        gout13 += gx[128] * gy[384] * gz[512];
                        gout14 += gx[0] * gy[512] * gz[512];
                        gout15 += gx[256] * gy[0] * gz[768];
                        gout16 += gx[128] * gy[0] * gz[896];
                        gout17 += gx[0] * gy[128] * gz[896];
                        break;
                    case 1:
                        gout0 += gx[896] * gy[128] * gz[0];
                        gout1 += gx[768] * gy[256] * gz[0];
                        gout2 += gx[768] * gy[0] * gz[256];
                        gout3 += gx[512] * gy[512] * gz[0];
                        gout4 += gx[384] * gy[640] * gz[0];
                        gout5 += gx[384] * gy[384] * gz[256];
                        gout6 += gx[512] * gy[128] * gz[384];
                        gout7 += gx[384] * gy[256] * gz[384];
                        gout8 += gx[384] * gy[0] * gz[640];
                        gout9 += gx[128] * gy[896] * gz[0];
                        gout10 += gx[0] * gy[1024] * gz[0];
                        gout11 += gx[0] * gy[768] * gz[256];
                        gout12 += gx[128] * gy[512] * gz[384];
                        gout13 += gx[0] * gy[640] * gz[384];
                        gout14 += gx[0] * gy[384] * gz[640];
                        gout15 += gx[128] * gy[128] * gz[768];
                        gout16 += gx[0] * gy[256] * gz[768];
                        gout17 += gx[0] * gy[0] * gz[1024];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    val += gout9 * dm[(l0+0)*nao+(k0+3)];
                    val += gout12 * dm[(l0+0)*nao+(k0+4)];
                    val += gout15 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout7 * dm[(l0+0)*nao+(k0+2)];
                    val += gout10 * dm[(l0+0)*nao+(k0+3)];
                    val += gout13 * dm[(l0+0)*nao+(k0+4)];
                    val += gout16 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout8 * dm[(l0+0)*nao+(k0+2)];
                    val += gout11 * dm[(l0+0)*nao+(k0+3)];
                    val += gout14 * dm[(l0+0)*nao+(k0+4)];
                    val += gout17 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    val += gout2 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+0)];
                    val += gout4 * dm[(j0+0)*nao+(i0+2)];
                    val += gout5 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+0)];
                    val += gout7 * dm[(j0+0)*nao+(i0+2)];
                    val += gout8 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+2)];
                    val += gout11 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+0)];
                    val += gout13 * dm[(j0+0)*nao+(i0+2)];
                    val += gout14 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(i0+0)];
                    val += gout16 * dm[(j0+0)*nao+(i0+2)];
                    val += gout17 * dm[(j0+0)*nao+(i0+4)];
                    atomicAdd(vj+(k0+5)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout3 * dm[(l0+0)*nao+(k0+1)];
                    val += gout6 * dm[(l0+0)*nao+(k0+2)];
                    val += gout9 * dm[(l0+0)*nao+(k0+3)];
                    val += gout12 * dm[(l0+0)*nao+(k0+4)];
                    val += gout15 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    val += gout7 * dm[(l0+0)*nao+(k0+2)];
                    val += gout10 * dm[(l0+0)*nao+(k0+3)];
                    val += gout13 * dm[(l0+0)*nao+(k0+4)];
                    val += gout16 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout8 * dm[(l0+0)*nao+(k0+2)];
                    val += gout11 * dm[(l0+0)*nao+(k0+3)];
                    val += gout14 * dm[(l0+0)*nao+(k0+4)];
                    val += gout17 * dm[(l0+0)*nao+(k0+5)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+3)];
                    val += gout2 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(i0+1)];
                    val += gout4 * dm[(j0+0)*nao+(i0+3)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(i0+1)];
                    val += gout7 * dm[(j0+0)*nao+(i0+3)];
                    val += gout8 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+1)];
                    val += gout10 * dm[(j0+0)*nao+(i0+3)];
                    val += gout11 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(i0+1)];
                    val += gout13 * dm[(j0+0)*nao+(i0+3)];
                    val += gout14 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(i0+1)];
                    val += gout16 * dm[(j0+0)*nao+(i0+3)];
                    val += gout17 * dm[(j0+0)*nao+(i0+5)];
                    atomicAdd(vj+(k0+5)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+1)];
                    val += gout6 * dm[(j0+0)*nao+(k0+2)];
                    val += gout9 * dm[(j0+0)*nao+(k0+3)];
                    val += gout12 * dm[(j0+0)*nao+(k0+4)];
                    val += gout15 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    val += gout10 * dm[(j0+0)*nao+(k0+3)];
                    val += gout13 * dm[(j0+0)*nao+(k0+4)];
                    val += gout16 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    val += gout11 * dm[(j0+0)*nao+(k0+3)];
                    val += gout14 * dm[(j0+0)*nao+(k0+4)];
                    val += gout17 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout3 * dm[(i0+0)*nao+(k0+1)];
                    val += gout6 * dm[(i0+0)*nao+(k0+2)];
                    val += gout9 * dm[(i0+0)*nao+(k0+3)];
                    val += gout12 * dm[(i0+0)*nao+(k0+4)];
                    val += gout15 * dm[(i0+0)*nao+(k0+5)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout4 * dm[(i0+2)*nao+(k0+1)];
                    val += gout7 * dm[(i0+2)*nao+(k0+2)];
                    val += gout10 * dm[(i0+2)*nao+(k0+3)];
                    val += gout13 * dm[(i0+2)*nao+(k0+4)];
                    val += gout16 * dm[(i0+2)*nao+(k0+5)];
                    val += gout2 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+4)*nao+(k0+1)];
                    val += gout8 * dm[(i0+4)*nao+(k0+2)];
                    val += gout11 * dm[(i0+4)*nao+(k0+3)];
                    val += gout14 * dm[(i0+4)*nao+(k0+4)];
                    val += gout17 * dm[(i0+4)*nao+(k0+5)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+3), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+4), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+5), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+3), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+4), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+5), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+3), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+4), val);
                    val = 0;
                    val += gout17 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+5), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+2)*nao+(l0+0)];
                    val += gout5 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+2)*nao+(l0+0)];
                    val += gout8 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+2)*nao+(l0+0)];
                    val += gout11 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+3), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+2)*nao+(l0+0)];
                    val += gout14 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+4), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(l0+0)];
                    val += gout16 * dm[(i0+2)*nao+(l0+0)];
                    val += gout17 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+5), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+0)*nao+(k0+1)];
                    val += gout6 * dm[(j0+0)*nao+(k0+2)];
                    val += gout9 * dm[(j0+0)*nao+(k0+3)];
                    val += gout12 * dm[(j0+0)*nao+(k0+4)];
                    val += gout15 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout7 * dm[(j0+0)*nao+(k0+2)];
                    val += gout10 * dm[(j0+0)*nao+(k0+3)];
                    val += gout13 * dm[(j0+0)*nao+(k0+4)];
                    val += gout16 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout8 * dm[(j0+0)*nao+(k0+2)];
                    val += gout11 * dm[(j0+0)*nao+(k0+3)];
                    val += gout14 * dm[(j0+0)*nao+(k0+4)];
                    val += gout17 * dm[(j0+0)*nao+(k0+5)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout3 * dm[(i0+1)*nao+(k0+1)];
                    val += gout6 * dm[(i0+1)*nao+(k0+2)];
                    val += gout9 * dm[(i0+1)*nao+(k0+3)];
                    val += gout12 * dm[(i0+1)*nao+(k0+4)];
                    val += gout15 * dm[(i0+1)*nao+(k0+5)];
                    val += gout1 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+3)*nao+(k0+1)];
                    val += gout7 * dm[(i0+3)*nao+(k0+2)];
                    val += gout10 * dm[(i0+3)*nao+(k0+3)];
                    val += gout13 * dm[(i0+3)*nao+(k0+4)];
                    val += gout16 * dm[(i0+3)*nao+(k0+5)];
                    val += gout2 * dm[(i0+5)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+1)];
                    val += gout8 * dm[(i0+5)*nao+(k0+2)];
                    val += gout11 * dm[(i0+5)*nao+(k0+3)];
                    val += gout14 * dm[(i0+5)*nao+(k0+4)];
                    val += gout17 * dm[(i0+5)*nao+(k0+5)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+3), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+4), val);
                    val = 0;
                    val += gout15 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+5), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+3), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+4), val);
                    val = 0;
                    val += gout16 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+5), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+3), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+4), val);
                    val = 0;
                    val += gout17 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+5), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout1 * dm[(i0+3)*nao+(l0+0)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout4 * dm[(i0+3)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+0)];
                    val += gout7 * dm[(i0+3)*nao+(l0+0)];
                    val += gout8 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout9 * dm[(i0+1)*nao+(l0+0)];
                    val += gout10 * dm[(i0+3)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+3), val);
                    val = 0;
                    val += gout12 * dm[(i0+1)*nao+(l0+0)];
                    val += gout13 * dm[(i0+3)*nao+(l0+0)];
                    val += gout14 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+4), val);
                    val = 0;
                    val += gout15 * dm[(i0+1)*nao+(l0+0)];
                    val += gout16 * dm[(i0+3)*nao+(l0+0)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+5), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2020(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2020(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 512;
    double *gy = gx + 768;
    double *gz = gy + 768;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 512;
                    rys_roots(2, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 768;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[128] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[256] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[384] = s2;
                        s1 = _gx[384];
                        s0 = _gx[256];
                        _gx[640] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[512] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[384] = s1 - xjxi * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[640] * gy[0] * gz[0];
                        gout1 += gx[512] * gy[0] * gz[128];
                        gout2 += gx[384] * gy[128] * gz[128];
                        gout3 += gx[256] * gy[384] * gz[0];
                        gout4 += gx[128] * gy[384] * gz[128];
                        gout5 += gx[0] * gy[512] * gz[128];
                        gout6 += gx[256] * gy[0] * gz[384];
                        gout7 += gx[128] * gy[0] * gz[512];
                        gout8 += gx[0] * gy[128] * gz[512];
                        break;
                    case 1:
                        gout0 += gx[512] * gy[128] * gz[0];
                        gout1 += gx[384] * gy[256] * gz[0];
                        gout2 += gx[384] * gy[0] * gz[256];
                        gout3 += gx[128] * gy[512] * gz[0];
                        gout4 += gx[0] * gy[640] * gz[0];
                        gout5 += gx[0] * gy[384] * gz[256];
                        gout6 += gx[128] * gy[128] * gz[384];
                        gout7 += gx[0] * gy[256] * gz[384];
                        gout8 += gx[0] * gy[0] * gz[640];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    val += gout2 * dm[(j0+0)*nao+(i0+4)];
                    val += gout3 * dm[(j0+1)*nao+(i0+0)];
                    val += gout4 * dm[(j0+1)*nao+(i0+2)];
                    val += gout5 * dm[(j0+1)*nao+(i0+4)];
                    val += gout6 * dm[(j0+2)*nao+(i0+0)];
                    val += gout7 * dm[(j0+2)*nao+(i0+2)];
                    val += gout8 * dm[(j0+2)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+3)];
                    val += gout2 * dm[(j0+0)*nao+(i0+5)];
                    val += gout3 * dm[(j0+1)*nao+(i0+1)];
                    val += gout4 * dm[(j0+1)*nao+(i0+3)];
                    val += gout5 * dm[(j0+1)*nao+(i0+5)];
                    val += gout6 * dm[(j0+2)*nao+(i0+1)];
                    val += gout7 * dm[(j0+2)*nao+(i0+3)];
                    val += gout8 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout2 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout4 * dm[(i0+2)*nao+(k0+0)];
                    val += gout5 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+2)*nao+(k0+0)];
                    val += gout8 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+2)*nao+(l0+0)];
                    val += gout5 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+2)*nao+(l0+0)];
                    val += gout8 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout1 * dm[(i0+3)*nao+(k0+0)];
                    val += gout2 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(k0+0)];
                    val += gout4 * dm[(i0+3)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(k0+0)];
                    val += gout7 * dm[(i0+3)*nao+(k0+0)];
                    val += gout8 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout1 * dm[(i0+3)*nao+(l0+0)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout4 * dm[(i0+3)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+0)];
                    val += gout7 * dm[(i0+3)*nao+(l0+0)];
                    val += gout8 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 384;
    double *gy = gx + 768;
    double *gz = gy + 768;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 64) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 64) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, 64, gout_id, 4);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 4) {
                        rw[ irys*2   *64] *= theta_fac;
                        rw[(irys*2+1)*64] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 384;
                    rys_roots(3, theta_rr, rw1, 64, gout_id, 4);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 64, gout_id, 4);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 4) {
                        rw[ irys*2   *64] *= theta_fac;
                        rw[(irys*2+1)*64] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gz[0] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 768;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[64] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[128] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[192] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
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
                        gout0 += gx[704] * gy[0] * gz[0];
                        gout1 += gx[576] * gy[64] * gz[64];
                        gout2 += gx[448] * gy[192] * gz[64];
                        gout3 += gx[512] * gy[0] * gz[192];
                        gout4 += gx[384] * gy[64] * gz[256];
                        gout5 += gx[256] * gy[384] * gz[64];
                        gout6 += gx[128] * gy[576] * gz[0];
                        gout7 += gx[0] * gy[640] * gz[64];
                        gout8 += gx[64] * gy[384] * gz[256];
                        gout9 += gx[320] * gy[0] * gz[384];
                        gout10 += gx[192] * gy[64] * gz[448];
                        gout11 += gx[64] * gy[192] * gz[448];
                        gout12 += gx[128] * gy[0] * gz[576];
                        gout13 += gx[0] * gy[64] * gz[640];
                        break;
                    case 1:
                        gout0 += gx[640] * gy[64] * gz[0];
                        gout1 += gx[576] * gy[0] * gz[128];
                        gout2 += gx[384] * gy[320] * gz[0];
                        gout3 += gx[448] * gy[64] * gz[192];
                        gout4 += gx[384] * gy[0] * gz[320];
                        gout5 += gx[192] * gy[512] * gz[0];
                        gout6 += gx[64] * gy[640] * gz[0];
                        gout7 += gx[0] * gy[576] * gz[128];
                        gout8 += gx[0] * gy[512] * gz[192];
                        gout9 += gx[256] * gy[64] * gz[384];
                        gout10 += gx[192] * gy[0] * gz[512];
                        gout11 += gx[0] * gy[320] * gz[384];
                        gout12 += gx[64] * gy[64] * gz[576];
                        gout13 += gx[0] * gy[0] * gz[704];
                        break;
                    case 2:
                        gout0 += gx[640] * gy[0] * gz[64];
                        gout1 += gx[512] * gy[192] * gz[0];
                        gout2 += gx[384] * gy[256] * gz[64];
                        gout3 += gx[448] * gy[0] * gz[256];
                        gout4 += gx[320] * gy[384] * gz[0];
                        gout5 += gx[192] * gy[448] * gz[64];
                        gout6 += gx[64] * gy[576] * gz[64];
                        gout7 += gx[128] * gy[384] * gz[192];
                        gout8 += gx[0] * gy[448] * gz[256];
                        gout9 += gx[256] * gy[0] * gz[448];
                        gout10 += gx[128] * gy[192] * gz[384];
                        gout11 += gx[0] * gy[256] * gz[448];
                        gout12 += gx[64] * gy[0] * gz[640];
                        break;
                    case 3:
                        gout0 += gx[576] * gy[128] * gz[0];
                        gout1 += gx[448] * gy[256] * gz[0];
                        gout2 += gx[384] * gy[192] * gz[128];
                        gout3 += gx[384] * gy[128] * gz[192];
                        gout4 += gx[256] * gy[448] * gz[0];
                        gout5 += gx[192] * gy[384] * gz[128];
                        gout6 += gx[0] * gy[704] * gz[0];
                        gout7 += gx[64] * gy[448] * gz[192];
                        gout8 += gx[0] * gy[384] * gz[320];
                        gout9 += gx[192] * gy[128] * gz[384];
                        gout10 += gx[64] * gy[256] * gz[384];
                        gout11 += gx[0] * gy[192] * gz[512];
                        gout12 += gx[0] * gy[128] * gz[576];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+4)];
                    val += gout2 * dm[(j0+1)*nao+(i0+2)];
                    val += gout3 * dm[(j0+2)*nao+(i0+0)];
                    val += gout4 * dm[(j0+2)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+2)];
                    val += gout6 * dm[(j0+1)*nao+(i0+0)];
                    val += gout7 * dm[(j0+1)*nao+(i0+4)];
                    val += gout8 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+0)];
                    val += gout10 * dm[(j0+0)*nao+(i0+4)];
                    val += gout11 * dm[(j0+1)*nao+(i0+2)];
                    val += gout12 * dm[(j0+2)*nao+(i0+0)];
                    val += gout13 * dm[(j0+2)*nao+(i0+4)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+5)];
                    val += gout2 * dm[(j0+1)*nao+(i0+3)];
                    val += gout3 * dm[(j0+2)*nao+(i0+1)];
                    val += gout4 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+3)];
                    val += gout6 * dm[(j0+1)*nao+(i0+1)];
                    val += gout7 * dm[(j0+1)*nao+(i0+5)];
                    val += gout8 * dm[(j0+2)*nao+(i0+3)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+1)];
                    val += gout10 * dm[(j0+0)*nao+(i0+5)];
                    val += gout11 * dm[(j0+1)*nao+(i0+3)];
                    val += gout12 * dm[(j0+2)*nao+(i0+1)];
                    val += gout13 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    case 2:
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+2)];
                    val += gout1 * dm[(j0+1)*nao+(i0+0)];
                    val += gout2 * dm[(j0+1)*nao+(i0+4)];
                    val += gout3 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+0)];
                    val += gout5 * dm[(j0+0)*nao+(i0+4)];
                    val += gout6 * dm[(j0+1)*nao+(i0+2)];
                    val += gout7 * dm[(j0+2)*nao+(i0+0)];
                    val += gout8 * dm[(j0+2)*nao+(i0+4)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+2)];
                    val += gout10 * dm[(j0+1)*nao+(i0+0)];
                    val += gout11 * dm[(j0+1)*nao+(i0+4)];
                    val += gout12 * dm[(j0+2)*nao+(i0+2)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    case 3:
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+3)];
                    val += gout1 * dm[(j0+1)*nao+(i0+1)];
                    val += gout2 * dm[(j0+1)*nao+(i0+5)];
                    val += gout3 * dm[(j0+2)*nao+(i0+3)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(i0+1)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+1)*nao+(i0+3)];
                    val += gout7 * dm[(j0+2)*nao+(i0+1)];
                    val += gout8 * dm[(j0+2)*nao+(i0+5)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(i0+3)];
                    val += gout10 * dm[(j0+1)*nao+(i0+1)];
                    val += gout11 * dm[(j0+1)*nao+(i0+5)];
                    val += gout12 * dm[(j0+2)*nao+(i0+3)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+2)];
                    val += gout6 * dm[(j0+1)*nao+(k0+1)];
                    val += gout3 * dm[(j0+2)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout2 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+2)];
                    val += gout7 * dm[(j0+1)*nao+(k0+1)];
                    val += gout4 * dm[(j0+2)*nao+(k0+0)];
                    val += gout13 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout9 * dm[(i0+0)*nao+(k0+2)];
                    val += gout5 * dm[(i0+2)*nao+(k0+1)];
                    val += gout1 * dm[(i0+4)*nao+(k0+0)];
                    val += gout10 * dm[(i0+4)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+1)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout11 * dm[(i0+2)*nao+(k0+2)];
                    val += gout7 * dm[(i0+4)*nao+(k0+1)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout12 * dm[(i0+0)*nao+(k0+2)];
                    val += gout8 * dm[(i0+2)*nao+(k0+1)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout13 * dm[(i0+4)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+2)];
                    val += gout6 * dm[(j0+1)*nao+(k0+1)];
                    val += gout3 * dm[(j0+2)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout2 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout10 * dm[(j0+0)*nao+(k0+2)];
                    val += gout7 * dm[(j0+1)*nao+(k0+1)];
                    val += gout4 * dm[(j0+2)*nao+(k0+0)];
                    val += gout13 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout9 * dm[(i0+1)*nao+(k0+2)];
                    val += gout5 * dm[(i0+3)*nao+(k0+1)];
                    val += gout1 * dm[(i0+5)*nao+(k0+0)];
                    val += gout10 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(k0+1)];
                    val += gout2 * dm[(i0+3)*nao+(k0+0)];
                    val += gout11 * dm[(i0+3)*nao+(k0+2)];
                    val += gout7 * dm[(i0+5)*nao+(k0+1)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(k0+0)];
                    val += gout12 * dm[(i0+1)*nao+(k0+2)];
                    val += gout8 * dm[(i0+3)*nao+(k0+1)];
                    val += gout4 * dm[(i0+5)*nao+(k0+0)];
                    val += gout13 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout1 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+1)*nao+(l0+0)];
                    val += gout10 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+0)];
                    val += gout7 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout4 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+1)*nao+(l0+0)];
                    val += gout13 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    case 2:
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout1 * dm[(j0+1)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+2)];
                    val += gout7 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+2)];
                    val += gout6 * dm[(j0+1)*nao+(k0+1)];
                    val += gout3 * dm[(j0+2)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout2 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(k0+1)];
                    val += gout0 * dm[(i0+2)*nao+(k0+0)];
                    val += gout9 * dm[(i0+2)*nao+(k0+2)];
                    val += gout5 * dm[(i0+4)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(k0+0)];
                    val += gout10 * dm[(i0+0)*nao+(k0+2)];
                    val += gout6 * dm[(i0+2)*nao+(k0+1)];
                    val += gout2 * dm[(i0+4)*nao+(k0+0)];
                    val += gout11 * dm[(i0+4)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(k0+1)];
                    val += gout3 * dm[(i0+2)*nao+(k0+0)];
                    val += gout12 * dm[(i0+2)*nao+(k0+2)];
                    val += gout8 * dm[(i0+4)*nao+(k0+1)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+0)*nao+(l0+0)];
                    val += gout5 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+0)*nao+(l0+0)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(l0+0)];
                    val += gout11 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(i0+0)*nao+(l0+0)];
                    val += gout8 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+2)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    case 3:
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+1)];
                    val += gout1 * dm[(j0+1)*nao+(k0+0)];
                    val += gout10 * dm[(j0+1)*nao+(k0+2)];
                    val += gout7 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+2)];
                    val += gout6 * dm[(j0+1)*nao+(k0+1)];
                    val += gout3 * dm[(j0+2)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout2 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+1)*nao+(k0+2)];
                    val += gout8 * dm[(j0+2)*nao+(k0+1)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+1)*nao+(k0+1)];
                    val += gout0 * dm[(i0+3)*nao+(k0+0)];
                    val += gout9 * dm[(i0+3)*nao+(k0+2)];
                    val += gout5 * dm[(i0+5)*nao+(k0+1)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout10 * dm[(i0+1)*nao+(k0+2)];
                    val += gout6 * dm[(i0+3)*nao+(k0+1)];
                    val += gout2 * dm[(i0+5)*nao+(k0+0)];
                    val += gout11 * dm[(i0+5)*nao+(k0+2)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(i0+1)*nao+(k0+1)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout12 * dm[(i0+3)*nao+(k0+2)];
                    val += gout8 * dm[(i0+5)*nao+(k0+1)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+1)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(i0+1)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout9 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+1)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(i0+1)*nao+(l0+0)];
                    val += gout8 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(i0+3)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+2), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2110(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_2200(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 768;
    double *gy = gx + 1152;
    double *gz = gy + 1152;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double gout15;
    double gout16;
    double gout17;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
        gout15 = 0;
        gout16 = 0;
        gout17 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 768;
                    rys_roots(3, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 1152;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
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
                        gout0 += gx[1024] * gy[0] * gz[0];
                        gout1 += gx[896] * gy[0] * gz[128];
                        gout2 += gx[768] * gy[128] * gz[128];
                        gout3 += gx[640] * gy[384] * gz[0];
                        gout4 += gx[512] * gy[384] * gz[128];
                        gout5 += gx[384] * gy[512] * gz[128];
                        gout6 += gx[640] * gy[0] * gz[384];
                        gout7 += gx[512] * gy[0] * gz[512];
                        gout8 += gx[384] * gy[128] * gz[512];
                        gout9 += gx[256] * gy[768] * gz[0];
                        gout10 += gx[128] * gy[768] * gz[128];
                        gout11 += gx[0] * gy[896] * gz[128];
                        gout12 += gx[256] * gy[384] * gz[384];
                        gout13 += gx[128] * gy[384] * gz[512];
                        gout14 += gx[0] * gy[512] * gz[512];
                        gout15 += gx[256] * gy[0] * gz[768];
                        gout16 += gx[128] * gy[0] * gz[896];
                        gout17 += gx[0] * gy[128] * gz[896];
                        break;
                    case 1:
                        gout0 += gx[896] * gy[128] * gz[0];
                        gout1 += gx[768] * gy[256] * gz[0];
                        gout2 += gx[768] * gy[0] * gz[256];
                        gout3 += gx[512] * gy[512] * gz[0];
                        gout4 += gx[384] * gy[640] * gz[0];
                        gout5 += gx[384] * gy[384] * gz[256];
                        gout6 += gx[512] * gy[128] * gz[384];
                        gout7 += gx[384] * gy[256] * gz[384];
                        gout8 += gx[384] * gy[0] * gz[640];
                        gout9 += gx[128] * gy[896] * gz[0];
                        gout10 += gx[0] * gy[1024] * gz[0];
                        gout11 += gx[0] * gy[768] * gz[256];
                        gout12 += gx[128] * gy[512] * gz[384];
                        gout13 += gx[0] * gy[640] * gz[384];
                        gout14 += gx[0] * gy[384] * gz[640];
                        gout15 += gx[128] * gy[128] * gz[768];
                        gout16 += gx[0] * gy[256] * gz[768];
                        gout17 += gx[0] * gy[0] * gz[1024];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+3), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+4), val);
                    val = 0;
                    val += gout15 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+5), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+3), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+4), val);
                    val = 0;
                    val += gout16 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+5), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+3), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+4), val);
                    val = 0;
                    val += gout17 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+5), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    val += gout2 * dm[(j0+0)*nao+(i0+4)];
                    val += gout3 * dm[(j0+1)*nao+(i0+0)];
                    val += gout4 * dm[(j0+1)*nao+(i0+2)];
                    val += gout5 * dm[(j0+1)*nao+(i0+4)];
                    val += gout6 * dm[(j0+2)*nao+(i0+0)];
                    val += gout7 * dm[(j0+2)*nao+(i0+2)];
                    val += gout8 * dm[(j0+2)*nao+(i0+4)];
                    val += gout9 * dm[(j0+3)*nao+(i0+0)];
                    val += gout10 * dm[(j0+3)*nao+(i0+2)];
                    val += gout11 * dm[(j0+3)*nao+(i0+4)];
                    val += gout12 * dm[(j0+4)*nao+(i0+0)];
                    val += gout13 * dm[(j0+4)*nao+(i0+2)];
                    val += gout14 * dm[(j0+4)*nao+(i0+4)];
                    val += gout15 * dm[(j0+5)*nao+(i0+0)];
                    val += gout16 * dm[(j0+5)*nao+(i0+2)];
                    val += gout17 * dm[(j0+5)*nao+(i0+4)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+3), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+4), val);
                    val = 0;
                    val += gout15 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+5), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+3), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+4), val);
                    val = 0;
                    val += gout16 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+5), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+3), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+4), val);
                    val = 0;
                    val += gout17 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+5), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+3)];
                    val += gout2 * dm[(j0+0)*nao+(i0+5)];
                    val += gout3 * dm[(j0+1)*nao+(i0+1)];
                    val += gout4 * dm[(j0+1)*nao+(i0+3)];
                    val += gout5 * dm[(j0+1)*nao+(i0+5)];
                    val += gout6 * dm[(j0+2)*nao+(i0+1)];
                    val += gout7 * dm[(j0+2)*nao+(i0+3)];
                    val += gout8 * dm[(j0+2)*nao+(i0+5)];
                    val += gout9 * dm[(j0+3)*nao+(i0+1)];
                    val += gout10 * dm[(j0+3)*nao+(i0+3)];
                    val += gout11 * dm[(j0+3)*nao+(i0+5)];
                    val += gout12 * dm[(j0+4)*nao+(i0+1)];
                    val += gout13 * dm[(j0+4)*nao+(i0+3)];
                    val += gout14 * dm[(j0+4)*nao+(i0+5)];
                    val += gout15 * dm[(j0+5)*nao+(i0+1)];
                    val += gout16 * dm[(j0+5)*nao+(i0+3)];
                    val += gout17 * dm[(j0+5)*nao+(i0+5)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    val += gout9 * dm[(j0+3)*nao+(k0+0)];
                    val += gout12 * dm[(j0+4)*nao+(k0+0)];
                    val += gout15 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    val += gout10 * dm[(j0+3)*nao+(k0+0)];
                    val += gout13 * dm[(j0+4)*nao+(k0+0)];
                    val += gout16 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    val += gout11 * dm[(j0+3)*nao+(k0+0)];
                    val += gout14 * dm[(j0+4)*nao+(k0+0)];
                    val += gout17 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout2 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(k0+0)];
                    val += gout4 * dm[(i0+2)*nao+(k0+0)];
                    val += gout5 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(k0+0)];
                    val += gout7 * dm[(i0+2)*nao+(k0+0)];
                    val += gout8 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(k0+0)];
                    val += gout10 * dm[(i0+2)*nao+(k0+0)];
                    val += gout11 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(k0+0)];
                    val += gout13 * dm[(i0+2)*nao+(k0+0)];
                    val += gout14 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(k0+0)];
                    val += gout16 * dm[(i0+2)*nao+(k0+0)];
                    val += gout17 * dm[(i0+4)*nao+(k0+0)];
                    atomicAdd(vk+(j0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    val += gout9 * dm[(j0+3)*nao+(l0+0)];
                    val += gout12 * dm[(j0+4)*nao+(l0+0)];
                    val += gout15 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    val += gout10 * dm[(j0+3)*nao+(l0+0)];
                    val += gout13 * dm[(j0+4)*nao+(l0+0)];
                    val += gout16 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    val += gout11 * dm[(j0+3)*nao+(l0+0)];
                    val += gout14 * dm[(j0+4)*nao+(l0+0)];
                    val += gout17 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+0)*nao+(l0+0)];
                    val += gout4 * dm[(i0+2)*nao+(l0+0)];
                    val += gout5 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+0)*nao+(l0+0)];
                    val += gout7 * dm[(i0+2)*nao+(l0+0)];
                    val += gout8 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+0)*nao+(l0+0)];
                    val += gout10 * dm[(i0+2)*nao+(l0+0)];
                    val += gout11 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+0)*nao+(l0+0)];
                    val += gout13 * dm[(i0+2)*nao+(l0+0)];
                    val += gout14 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout15 * dm[(i0+0)*nao+(l0+0)];
                    val += gout16 * dm[(i0+2)*nao+(l0+0)];
                    val += gout17 * dm[(i0+4)*nao+(l0+0)];
                    atomicAdd(vk+(j0+5)*nao+(k0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout3 * dm[(j0+1)*nao+(k0+0)];
                    val += gout6 * dm[(j0+2)*nao+(k0+0)];
                    val += gout9 * dm[(j0+3)*nao+(k0+0)];
                    val += gout12 * dm[(j0+4)*nao+(k0+0)];
                    val += gout15 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout4 * dm[(j0+1)*nao+(k0+0)];
                    val += gout7 * dm[(j0+2)*nao+(k0+0)];
                    val += gout10 * dm[(j0+3)*nao+(k0+0)];
                    val += gout13 * dm[(j0+4)*nao+(k0+0)];
                    val += gout16 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout8 * dm[(j0+2)*nao+(k0+0)];
                    val += gout11 * dm[(j0+3)*nao+(k0+0)];
                    val += gout14 * dm[(j0+4)*nao+(k0+0)];
                    val += gout17 * dm[(j0+5)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout1 * dm[(i0+3)*nao+(k0+0)];
                    val += gout2 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(k0+0)];
                    val += gout4 * dm[(i0+3)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(k0+0)];
                    val += gout7 * dm[(i0+3)*nao+(k0+0)];
                    val += gout8 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+1)*nao+(k0+0)];
                    val += gout10 * dm[(i0+3)*nao+(k0+0)];
                    val += gout11 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+1)*nao+(k0+0)];
                    val += gout13 * dm[(i0+3)*nao+(k0+0)];
                    val += gout14 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout15 * dm[(i0+1)*nao+(k0+0)];
                    val += gout16 * dm[(i0+3)*nao+(k0+0)];
                    val += gout17 * dm[(i0+5)*nao+(k0+0)];
                    atomicAdd(vk+(j0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout3 * dm[(j0+1)*nao+(l0+0)];
                    val += gout6 * dm[(j0+2)*nao+(l0+0)];
                    val += gout9 * dm[(j0+3)*nao+(l0+0)];
                    val += gout12 * dm[(j0+4)*nao+(l0+0)];
                    val += gout15 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout4 * dm[(j0+1)*nao+(l0+0)];
                    val += gout7 * dm[(j0+2)*nao+(l0+0)];
                    val += gout10 * dm[(j0+3)*nao+(l0+0)];
                    val += gout13 * dm[(j0+4)*nao+(l0+0)];
                    val += gout16 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout8 * dm[(j0+2)*nao+(l0+0)];
                    val += gout11 * dm[(j0+3)*nao+(l0+0)];
                    val += gout14 * dm[(j0+4)*nao+(l0+0)];
                    val += gout17 * dm[(j0+5)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout1 * dm[(i0+3)*nao+(l0+0)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(i0+1)*nao+(l0+0)];
                    val += gout4 * dm[(i0+3)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(i0+1)*nao+(l0+0)];
                    val += gout7 * dm[(i0+3)*nao+(l0+0)];
                    val += gout8 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(i0+1)*nao+(l0+0)];
                    val += gout10 * dm[(i0+3)*nao+(l0+0)];
                    val += gout11 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout12 * dm[(i0+1)*nao+(l0+0)];
                    val += gout13 * dm[(i0+3)*nao+(l0+0)];
                    val += gout14 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout15 * dm[(i0+1)*nao+(l0+0)];
                    val += gout16 * dm[(i0+3)*nao+(l0+0)];
                    val += gout17 * dm[(i0+5)*nao+(l0+0)];
                    atomicAdd(vk+(j0+5)*nao+(k0+0), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_2200(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_2200(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_3000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 1024;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 256) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 256) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, 256, gout_id, 1);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 1024;
                    rys_roots(2, theta_rr, rw1, 256, gout_id, 1);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, 256, gout_id, 1);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; irys += 1) {
                        rw[ irys*2   *256] *= theta_fac;
                        rw[(irys*2+1)*256] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*512];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    for (int n = gout_id; n < 3; n += 1) {
                        if (n == 2) {
                            gz[0] = rw[irys*512+256];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[256] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[512] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[768] = s2;
                    }
                    __syncthreads();
                        gout0 += gx[768] * gy[0] * gz[0];
                        gout1 += gx[512] * gy[256] * gz[0];
                        gout2 += gx[512] * gy[0] * gz[256];
                        gout3 += gx[256] * gy[512] * gz[0];
                        gout4 += gx[256] * gy[256] * gz[256];
                        gout5 += gx[256] * gy[0] * gz[512];
                        gout6 += gx[0] * gy[768] * gz[0];
                        gout7 += gx[0] * gy[512] * gz[256];
                        gout8 += gx[0] * gy[256] * gz[512];
                        gout9 += gx[0] * gy[0] * gz[768];
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+1)];
                    val += gout2 * dm[(j0+0)*nao+(i0+2)];
                    val += gout3 * dm[(j0+0)*nao+(i0+3)];
                    val += gout4 * dm[(j0+0)*nao+(i0+4)];
                    val += gout5 * dm[(j0+0)*nao+(i0+5)];
                    val += gout6 * dm[(j0+0)*nao+(i0+6)];
                    val += gout7 * dm[(j0+0)*nao+(i0+7)];
                    val += gout8 * dm[(j0+0)*nao+(i0+8)];
                    val += gout9 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    vj += nao * nao;
                }
                if (do_k) {
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(k0+0)];
                    atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+1)*nao+(k0+0)];
                    val += gout2 * dm[(i0+2)*nao+(k0+0)];
                    val += gout3 * dm[(i0+3)*nao+(k0+0)];
                    val += gout4 * dm[(i0+4)*nao+(k0+0)];
                    val += gout5 * dm[(i0+5)*nao+(k0+0)];
                    val += gout6 * dm[(i0+6)*nao+(k0+0)];
                    val += gout7 * dm[(i0+7)*nao+(k0+0)];
                    val += gout8 * dm[(i0+8)*nao+(k0+0)];
                    val += gout9 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+1)*nao+(l0+0)];
                    val += gout2 * dm[(i0+2)*nao+(l0+0)];
                    val += gout3 * dm[(i0+3)*nao+(l0+0)];
                    val += gout4 * dm[(i0+4)*nao+(l0+0)];
                    val += gout5 * dm[(i0+5)*nao+(l0+0)];
                    val += gout6 * dm[(i0+6)*nao+(l0+0)];
                    val += gout7 * dm[(i0+7)*nao+(l0+0)];
                    val += gout8 * dm[(i0+8)*nao+(l0+0)];
                    val += gout9 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_3000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_3000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_3010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 768;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 768;
                    rys_roots(3, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[128] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[256] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[384] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[512] = s1;
                        s0 = _gx[128];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[640] = s1;
                        s0 = _gx[256];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[128];
                        _gx[768] = s1;
                        s0 = _gx[384];
                        s1 = cpx * s0;
                        s1 += 3 * b00 * _gx[256];
                        _gx[896] = s1;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[896] * gy[0] * gz[0];
                        gout1 += gx[768] * gy[0] * gz[128];
                        gout2 += gx[640] * gy[128] * gz[128];
                        gout3 += gx[512] * gy[384] * gz[0];
                        gout4 += gx[512] * gy[128] * gz[256];
                        gout5 += gx[384] * gy[512] * gz[0];
                        gout6 += gx[256] * gy[512] * gz[128];
                        gout7 += gx[128] * gy[640] * gz[128];
                        gout8 += gx[0] * gy[896] * gz[0];
                        gout9 += gx[0] * gy[640] * gz[256];
                        gout10 += gx[384] * gy[0] * gz[512];
                        gout11 += gx[256] * gy[0] * gz[640];
                        gout12 += gx[128] * gy[128] * gz[640];
                        gout13 += gx[0] * gy[384] * gz[512];
                        gout14 += gx[0] * gy[128] * gz[768];
                        break;
                    case 1:
                        gout0 += gx[768] * gy[128] * gz[0];
                        gout1 += gx[640] * gy[256] * gz[0];
                        gout2 += gx[640] * gy[0] * gz[256];
                        gout3 += gx[512] * gy[256] * gz[128];
                        gout4 += gx[512] * gy[0] * gz[384];
                        gout5 += gx[256] * gy[640] * gz[0];
                        gout6 += gx[128] * gy[768] * gz[0];
                        gout7 += gx[128] * gy[512] * gz[256];
                        gout8 += gx[0] * gy[768] * gz[128];
                        gout9 += gx[0] * gy[512] * gz[384];
                        gout10 += gx[256] * gy[128] * gz[512];
                        gout11 += gx[128] * gy[256] * gz[512];
                        gout12 += gx[128] * gy[0] * gz[768];
                        gout13 += gx[0] * gy[256] * gz[640];
                        gout14 += gx[0] * gy[0] * gz[896];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+6)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+1)];
                    val += gout14 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+8)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    val += gout2 * dm[(j0+0)*nao+(i0+4)];
                    val += gout3 * dm[(j0+0)*nao+(i0+6)];
                    val += gout4 * dm[(j0+0)*nao+(i0+8)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+0)];
                    val += gout6 * dm[(j0+0)*nao+(i0+2)];
                    val += gout7 * dm[(j0+0)*nao+(i0+4)];
                    val += gout8 * dm[(j0+0)*nao+(i0+6)];
                    val += gout9 * dm[(j0+0)*nao+(i0+8)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(i0+0)];
                    val += gout11 * dm[(j0+0)*nao+(i0+2)];
                    val += gout12 * dm[(j0+0)*nao+(i0+4)];
                    val += gout13 * dm[(j0+0)*nao+(i0+6)];
                    val += gout14 * dm[(j0+0)*nao+(i0+8)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    val += gout5 * dm[(l0+0)*nao+(k0+1)];
                    val += gout10 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    val += gout6 * dm[(l0+0)*nao+(k0+1)];
                    val += gout11 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    val += gout7 * dm[(l0+0)*nao+(k0+1)];
                    val += gout12 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    val += gout8 * dm[(l0+0)*nao+(k0+1)];
                    val += gout13 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+7)*nao+(j0+0), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    val += gout9 * dm[(l0+0)*nao+(k0+1)];
                    val += gout14 * dm[(l0+0)*nao+(k0+2)];
                    atomicAdd(vj+(i0+9)*nao+(j0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+3)];
                    val += gout2 * dm[(j0+0)*nao+(i0+5)];
                    val += gout3 * dm[(j0+0)*nao+(i0+7)];
                    val += gout4 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(i0+1)];
                    val += gout6 * dm[(j0+0)*nao+(i0+3)];
                    val += gout7 * dm[(j0+0)*nao+(i0+5)];
                    val += gout8 * dm[(j0+0)*nao+(i0+7)];
                    val += gout9 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(i0+1)];
                    val += gout11 * dm[(j0+0)*nao+(i0+3)];
                    val += gout12 * dm[(j0+0)*nao+(i0+5)];
                    val += gout13 * dm[(j0+0)*nao+(i0+7)];
                    val += gout14 * dm[(j0+0)*nao+(i0+9)];
                    atomicAdd(vj+(k0+2)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout10 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    val += gout11 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+1)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout14 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout5 * dm[(i0+0)*nao+(k0+1)];
                    val += gout10 * dm[(i0+0)*nao+(k0+2)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout6 * dm[(i0+2)*nao+(k0+1)];
                    val += gout11 * dm[(i0+2)*nao+(k0+2)];
                    val += gout2 * dm[(i0+4)*nao+(k0+0)];
                    val += gout7 * dm[(i0+4)*nao+(k0+1)];
                    val += gout12 * dm[(i0+4)*nao+(k0+2)];
                    val += gout3 * dm[(i0+6)*nao+(k0+0)];
                    val += gout8 * dm[(i0+6)*nao+(k0+1)];
                    val += gout13 * dm[(i0+6)*nao+(k0+2)];
                    val += gout4 * dm[(i0+8)*nao+(k0+0)];
                    val += gout9 * dm[(i0+8)*nao+(k0+1)];
                    val += gout14 * dm[(i0+8)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    val += gout3 * dm[(i0+6)*nao+(l0+0)];
                    val += gout4 * dm[(i0+8)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+0)*nao+(l0+0)];
                    val += gout6 * dm[(i0+2)*nao+(l0+0)];
                    val += gout7 * dm[(i0+4)*nao+(l0+0)];
                    val += gout8 * dm[(i0+6)*nao+(l0+0)];
                    val += gout9 * dm[(i0+8)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+0)];
                    val += gout12 * dm[(i0+4)*nao+(l0+0)];
                    val += gout13 * dm[(i0+6)*nao+(l0+0)];
                    val += gout14 * dm[(i0+8)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+0)*nao+(k0+1)];
                    val += gout10 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+0)*nao+(k0+1)];
                    val += gout11 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+0)*nao+(k0+1)];
                    val += gout12 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+0)*nao+(k0+1)];
                    val += gout13 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+0)*nao+(k0+1)];
                    val += gout14 * dm[(j0+0)*nao+(k0+2)];
                    atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout5 * dm[(i0+1)*nao+(k0+1)];
                    val += gout10 * dm[(i0+1)*nao+(k0+2)];
                    val += gout1 * dm[(i0+3)*nao+(k0+0)];
                    val += gout6 * dm[(i0+3)*nao+(k0+1)];
                    val += gout11 * dm[(i0+3)*nao+(k0+2)];
                    val += gout2 * dm[(i0+5)*nao+(k0+0)];
                    val += gout7 * dm[(i0+5)*nao+(k0+1)];
                    val += gout12 * dm[(i0+5)*nao+(k0+2)];
                    val += gout3 * dm[(i0+7)*nao+(k0+0)];
                    val += gout8 * dm[(i0+7)*nao+(k0+1)];
                    val += gout13 * dm[(i0+7)*nao+(k0+2)];
                    val += gout4 * dm[(i0+9)*nao+(k0+0)];
                    val += gout9 * dm[(i0+9)*nao+(k0+1)];
                    val += gout14 * dm[(i0+9)*nao+(k0+2)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout6 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                    val = 0;
                    val += gout11 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout7 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                    val = 0;
                    val += gout12 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                    val = 0;
                    val += gout8 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+1), val);
                    val = 0;
                    val += gout13 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+2), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                    val = 0;
                    val += gout9 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+1), val);
                    val = 0;
                    val += gout14 * dm[(j0+0)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+2), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout1 * dm[(i0+3)*nao+(l0+0)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    val += gout3 * dm[(i0+7)*nao+(l0+0)];
                    val += gout4 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+1)*nao+(l0+0)];
                    val += gout6 * dm[(i0+3)*nao+(l0+0)];
                    val += gout7 * dm[(i0+5)*nao+(l0+0)];
                    val += gout8 * dm[(i0+7)*nao+(l0+0)];
                    val += gout9 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                    val = 0;
                    val += gout10 * dm[(i0+1)*nao+(l0+0)];
                    val += gout11 * dm[(i0+3)*nao+(l0+0)];
                    val += gout12 * dm[(i0+5)*nao+(l0+0)];
                    val += gout13 * dm[(i0+7)*nao+(l0+0)];
                    val += gout14 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_3010(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_3010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_jk_3100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    extern __shared__ double cache_cicj[];
    double *rw = cache_cicj + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 768;
    double *gy = gx + 1024;
    double *gz = gy + 1024;

    if (gout_id == 0) {
        gx[0] = 1.;

        for (int n = sq_id; n < iprim*jprim*TILE2; n += 128) {
            int ijp = n / TILE2;
            int sh_ij = n % TILE2;
            int ish = ish0 + sh_ij / TILE;
            int jsh = jsh0 + sh_ij % TILE;
            int ip = ijp / jprim;
            int jp = ijp % jprim;
            double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
            double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
            double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
            double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
            double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            cache_cicj[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
        }
    }
    double gout0;
    double gout1;
    double gout2;
    double gout3;
    double gout4;
    double gout5;
    double gout6;
    double gout7;
    double gout8;
    double gout9;
    double gout10;
    double gout11;
    double gout12;
    double gout13;
    double gout14;
    double s0, s1, s2;
    double val;
    double *dm, *vj, *vk;
    double Rpq[3];

    for (int task0 = 0; task0 < ntasks; task0 += 128) {
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
        int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= .5;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        
        gout0 = 0;
        gout1 = 0;
        gout2 = 0;
        gout3 = 0;
        gout4 = 0;
        gout5 = 0;
        gout6 = 0;
        gout7 = 0;
        gout8 = 0;
        gout9 = 0;
        gout10 = 0;
        gout11 = 0;
        gout12 = 0;
        gout13 = 0;
        gout14 = 0;
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
            double theta_kl = ak * al / akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = cache_cicj[sh_ij+ijp*TILE2];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    gy[0] = fac;
                }
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa;
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                double omega = env[PTR_RANGE_OMEGA];
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, 128, gout_id, 2);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 768;
                    rys_roots(3, theta_rr, rw1, 128, gout_id, 2);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, 128, gout_id, 2);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; irys += 2) {
                        rw[ irys*2   *128] *= theta_fac;
                        rw[(irys*2+1)*128] *= sqrt_theta_fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*256];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    for (int n = gout_id; n < 3; n += 2) {
                        if (n == 2) {
                            gz[0] = rw[irys*256+128];
                        }
                        double *_gx = gx + n * 1024;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
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
                        _gx[896] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[256];
                        _gx[768] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[640] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[512] = s1 - xjxi * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        gout0 += gx[896] * gy[0] * gz[0];
                        gout1 += gx[768] * gy[0] * gz[128];
                        gout2 += gx[640] * gy[128] * gz[128];
                        gout3 += gx[512] * gy[384] * gz[0];
                        gout4 += gx[512] * gy[128] * gz[256];
                        gout5 += gx[384] * gy[512] * gz[0];
                        gout6 += gx[256] * gy[512] * gz[128];
                        gout7 += gx[128] * gy[640] * gz[128];
                        gout8 += gx[0] * gy[896] * gz[0];
                        gout9 += gx[0] * gy[640] * gz[256];
                        gout10 += gx[384] * gy[0] * gz[512];
                        gout11 += gx[256] * gy[0] * gz[640];
                        gout12 += gx[128] * gy[128] * gz[640];
                        gout13 += gx[0] * gy[384] * gz[512];
                        gout14 += gx[0] * gy[128] * gz[768];
                        break;
                    case 1:
                        gout0 += gx[768] * gy[128] * gz[0];
                        gout1 += gx[640] * gy[256] * gz[0];
                        gout2 += gx[640] * gy[0] * gz[256];
                        gout3 += gx[512] * gy[256] * gz[128];
                        gout4 += gx[512] * gy[0] * gz[384];
                        gout5 += gx[256] * gy[640] * gz[0];
                        gout6 += gx[128] * gy[768] * gz[0];
                        gout7 += gx[128] * gy[512] * gz[256];
                        gout8 += gx[0] * gy[768] * gz[128];
                        gout9 += gx[0] * gy[512] * gz[384];
                        gout10 += gx[256] * gy[128] * gz[512];
                        gout11 += gx[128] * gy[256] * gz[512];
                        gout12 += gx[128] * gy[0] * gz[768];
                        gout13 += gx[0] * gy[256] * gz[640];
                        gout14 += gx[0] * gy[0] * gz[896];
                        break;
                    }
                }
            }
        }
        if (task_id < ntasks) {
            dm = jk.dm;
            vj = jk.vj;
            vk = jk.vk;
            int do_j = vj != NULL;
            int do_k = vk != NULL;
            for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
                if (do_j) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+1), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+0)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+1), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+2)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+1), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+4)*nao+(j0+2), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+1), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+6)*nao+(j0+2), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+1), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+8)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+0)];
                    val += gout1 * dm[(j0+0)*nao+(i0+2)];
                    val += gout2 * dm[(j0+0)*nao+(i0+4)];
                    val += gout3 * dm[(j0+0)*nao+(i0+6)];
                    val += gout4 * dm[(j0+0)*nao+(i0+8)];
                    val += gout5 * dm[(j0+1)*nao+(i0+0)];
                    val += gout6 * dm[(j0+1)*nao+(i0+2)];
                    val += gout7 * dm[(j0+1)*nao+(i0+4)];
                    val += gout8 * dm[(j0+1)*nao+(i0+6)];
                    val += gout9 * dm[(j0+1)*nao+(i0+8)];
                    val += gout10 * dm[(j0+2)*nao+(i0+0)];
                    val += gout11 * dm[(j0+2)*nao+(i0+2)];
                    val += gout12 * dm[(j0+2)*nao+(i0+4)];
                    val += gout13 * dm[(j0+2)*nao+(i0+6)];
                    val += gout14 * dm[(j0+2)*nao+(i0+8)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+0), val);
                    val = 0;
                    val += gout5 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+1), val);
                    val = 0;
                    val += gout10 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+1)*nao+(j0+2), val);
                    val = 0;
                    val += gout1 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+0), val);
                    val = 0;
                    val += gout6 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+1), val);
                    val = 0;
                    val += gout11 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+3)*nao+(j0+2), val);
                    val = 0;
                    val += gout2 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+0), val);
                    val = 0;
                    val += gout7 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+1), val);
                    val = 0;
                    val += gout12 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+5)*nao+(j0+2), val);
                    val = 0;
                    val += gout3 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+0), val);
                    val = 0;
                    val += gout8 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+1), val);
                    val = 0;
                    val += gout13 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+7)*nao+(j0+2), val);
                    val = 0;
                    val += gout4 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+0), val);
                    val = 0;
                    val += gout9 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+1), val);
                    val = 0;
                    val += gout14 * dm[(l0+0)*nao+(k0+0)];
                    atomicAdd(vj+(i0+9)*nao+(j0+2), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(i0+1)];
                    val += gout1 * dm[(j0+0)*nao+(i0+3)];
                    val += gout2 * dm[(j0+0)*nao+(i0+5)];
                    val += gout3 * dm[(j0+0)*nao+(i0+7)];
                    val += gout4 * dm[(j0+0)*nao+(i0+9)];
                    val += gout5 * dm[(j0+1)*nao+(i0+1)];
                    val += gout6 * dm[(j0+1)*nao+(i0+3)];
                    val += gout7 * dm[(j0+1)*nao+(i0+5)];
                    val += gout8 * dm[(j0+1)*nao+(i0+7)];
                    val += gout9 * dm[(j0+1)*nao+(i0+9)];
                    val += gout10 * dm[(j0+2)*nao+(i0+1)];
                    val += gout11 * dm[(j0+2)*nao+(i0+3)];
                    val += gout12 * dm[(j0+2)*nao+(i0+5)];
                    val += gout13 * dm[(j0+2)*nao+(i0+7)];
                    val += gout14 * dm[(j0+2)*nao+(i0+9)];
                    atomicAdd(vj+(k0+0)*nao+(l0+0), val);
                    break;
                    }
                    vj += nao * nao;
                }
                if (do_k) {
                    switch (gout_id) {
                    case 0:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout10 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+1)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+1)*nao+(k0+0)];
                    val += gout13 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+1)*nao+(k0+0)];
                    val += gout14 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(k0+0)];
                    val += gout1 * dm[(i0+2)*nao+(k0+0)];
                    val += gout2 * dm[(i0+4)*nao+(k0+0)];
                    val += gout3 * dm[(i0+6)*nao+(k0+0)];
                    val += gout4 * dm[(i0+8)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+0)*nao+(k0+0)];
                    val += gout6 * dm[(i0+2)*nao+(k0+0)];
                    val += gout7 * dm[(i0+4)*nao+(k0+0)];
                    val += gout8 * dm[(i0+6)*nao+(k0+0)];
                    val += gout9 * dm[(i0+8)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(k0+0)];
                    val += gout11 * dm[(i0+2)*nao+(k0+0)];
                    val += gout12 * dm[(i0+4)*nao+(k0+0)];
                    val += gout13 * dm[(i0+6)*nao+(k0+0)];
                    val += gout14 * dm[(i0+8)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout10 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    val += gout11 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+1)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+1)*nao+(l0+0)];
                    val += gout14 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+0)*nao+(l0+0)];
                    val += gout1 * dm[(i0+2)*nao+(l0+0)];
                    val += gout2 * dm[(i0+4)*nao+(l0+0)];
                    val += gout3 * dm[(i0+6)*nao+(l0+0)];
                    val += gout4 * dm[(i0+8)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+0)*nao+(l0+0)];
                    val += gout6 * dm[(i0+2)*nao+(l0+0)];
                    val += gout7 * dm[(i0+4)*nao+(l0+0)];
                    val += gout8 * dm[(i0+6)*nao+(l0+0)];
                    val += gout9 * dm[(i0+8)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+0)*nao+(l0+0)];
                    val += gout11 * dm[(i0+2)*nao+(l0+0)];
                    val += gout12 * dm[(i0+4)*nao+(l0+0)];
                    val += gout13 * dm[(i0+6)*nao+(l0+0)];
                    val += gout14 * dm[(i0+8)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    break;
                    case 1:
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(k0+0)];
                    val += gout5 * dm[(j0+1)*nao+(k0+0)];
                    val += gout10 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(k0+0)];
                    val += gout6 * dm[(j0+1)*nao+(k0+0)];
                    val += gout11 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(k0+0)];
                    val += gout7 * dm[(j0+1)*nao+(k0+0)];
                    val += gout12 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(k0+0)];
                    val += gout8 * dm[(j0+1)*nao+(k0+0)];
                    val += gout13 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(k0+0)];
                    val += gout9 * dm[(j0+1)*nao+(k0+0)];
                    val += gout14 * dm[(j0+2)*nao+(k0+0)];
                    atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(k0+0)];
                    val += gout1 * dm[(i0+3)*nao+(k0+0)];
                    val += gout2 * dm[(i0+5)*nao+(k0+0)];
                    val += gout3 * dm[(i0+7)*nao+(k0+0)];
                    val += gout4 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+1)*nao+(k0+0)];
                    val += gout6 * dm[(i0+3)*nao+(k0+0)];
                    val += gout7 * dm[(i0+5)*nao+(k0+0)];
                    val += gout8 * dm[(i0+7)*nao+(k0+0)];
                    val += gout9 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+1)*nao+(k0+0)];
                    val += gout11 * dm[(i0+3)*nao+(k0+0)];
                    val += gout12 * dm[(i0+5)*nao+(k0+0)];
                    val += gout13 * dm[(i0+7)*nao+(k0+0)];
                    val += gout14 * dm[(i0+9)*nao+(k0+0)];
                    atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                    val = 0;
                    val += gout0 * dm[(j0+0)*nao+(l0+0)];
                    val += gout5 * dm[(j0+1)*nao+(l0+0)];
                    val += gout10 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout1 * dm[(j0+0)*nao+(l0+0)];
                    val += gout6 * dm[(j0+1)*nao+(l0+0)];
                    val += gout11 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                    val = 0;
                    val += gout2 * dm[(j0+0)*nao+(l0+0)];
                    val += gout7 * dm[(j0+1)*nao+(l0+0)];
                    val += gout12 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                    val = 0;
                    val += gout3 * dm[(j0+0)*nao+(l0+0)];
                    val += gout8 * dm[(j0+1)*nao+(l0+0)];
                    val += gout13 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                    val = 0;
                    val += gout4 * dm[(j0+0)*nao+(l0+0)];
                    val += gout9 * dm[(j0+1)*nao+(l0+0)];
                    val += gout14 * dm[(j0+2)*nao+(l0+0)];
                    atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                    val = 0;
                    val += gout0 * dm[(i0+1)*nao+(l0+0)];
                    val += gout1 * dm[(i0+3)*nao+(l0+0)];
                    val += gout2 * dm[(i0+5)*nao+(l0+0)];
                    val += gout3 * dm[(i0+7)*nao+(l0+0)];
                    val += gout4 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                    val = 0;
                    val += gout5 * dm[(i0+1)*nao+(l0+0)];
                    val += gout6 * dm[(i0+3)*nao+(l0+0)];
                    val += gout7 * dm[(i0+5)*nao+(l0+0)];
                    val += gout8 * dm[(i0+7)*nao+(l0+0)];
                    val += gout9 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                    val = 0;
                    val += gout10 * dm[(i0+1)*nao+(l0+0)];
                    val += gout11 * dm[(i0+3)*nao+(l0+0)];
                    val += gout12 * dm[(i0+5)*nao+(l0+0)];
                    val += gout13 * dm[(i0+7)*nao+(l0+0)];
                    val += gout14 * dm[(i0+9)*nao+(l0+0)];
                    atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                    break;
                    }
                    vk += nao * nao;
                }
                dm += nao * nao;
            }
        }
    }
}
__global__
void rys_jk_3100(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                    batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas = envs.nbas;
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_jk_3100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                    ShellQuartet *pool, uint32_t *batch_head,
                    int *scheme, int workers, double omega)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int ijkl = li*125 + lj*25 + lk*5 + ll;
    int nroots = (li + lj + lk + ll) / 2 + 1;
    int g_size = bounds->stride_l * (bounds->ll + 1);
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ij_prims = iprim * jprim;
    int buflen = ij_prims*TILE2;
    int nsq_per_block = 256;
    int gout_stride = 1;

    switch (ijkl) {
    case 125:
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 130:
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 131:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 150:
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 155:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 156:
        nsq_per_block = 64;
        gout_stride = 4;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 250:
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 255:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 256:
        nsq_per_block = 64;
        gout_stride = 4;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 260:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 275:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 280:
        nsq_per_block = 64;
        gout_stride = 4;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 300:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 375:
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 380:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    case 400:
        nsq_per_block = 128;
        gout_stride = 2;
        buflen += g_size * 3 * nsq_per_block;
        break;
    }

#if CUDA_VERSION >= 12040
    switch (ijkl) {
    case 0: nsq_per_block *= 2; break;
    }
#endif

    dim3 threads(nsq_per_block, gout_stride);
    buflen += nroots*2 * nsq_per_block;
    if (omega < 0) {
        buflen += nroots*2 * nsq_per_block;
    }
    switch (ijkl) {
    case 0:
        buflen += ij_prims*TILE2*3;
        rys_jk_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 125:
        cudaFuncSetAttribute(rys_jk_1000, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 130:
        cudaFuncSetAttribute(rys_jk_1010, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_1010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 131:
        cudaFuncSetAttribute(rys_jk_1011, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_1011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 150:
        cudaFuncSetAttribute(rys_jk_1100, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_1100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 155:
        cudaFuncSetAttribute(rys_jk_1110, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_1110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 156:
        cudaFuncSetAttribute(rys_jk_1111, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_1111<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 250:
        cudaFuncSetAttribute(rys_jk_2000, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_2000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 255:
        cudaFuncSetAttribute(rys_jk_2010, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_2010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 256:
        cudaFuncSetAttribute(rys_jk_2011, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_2011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 260:
        cudaFuncSetAttribute(rys_jk_2020, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_2020<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 275:
        cudaFuncSetAttribute(rys_jk_2100, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_2100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 280:
        cudaFuncSetAttribute(rys_jk_2110, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_2110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 300:
        cudaFuncSetAttribute(rys_jk_2200, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_2200<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 375:
        cudaFuncSetAttribute(rys_jk_3000, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_3000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 380:
        cudaFuncSetAttribute(rys_jk_3010, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_3010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 400:
        cudaFuncSetAttribute(rys_jk_3100, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_jk_3100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    return 1;
}
